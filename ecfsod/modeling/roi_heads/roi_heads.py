import torch
import logging
import numpy as np
from .cpln import CPLN
from typing import Dict
from .box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, ECFSODOutputs
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")

logger = logging.getLogger(__name__)

def build_roi_heads(cfg, input_shape):
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

class ROIHeads(torch.nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()
        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            matched_iou = match_quality_matrix[matched_idxs, torch.arange(match_quality_matrix.shape[1])]
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.ious = matched_iou[sampled_idxs]
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        raise NotImplementedError()

@ROI_HEADS_REGISTRY.register()
class ECFSODROIHeads(ROIHeads):
    def __init__(self, cfg, input_shape):
        super(ECFSODROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        in_channels = [self.feature_channels[f] for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.box_head = build_box_head(cfg,ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution))
        self.input_size = self.box_head.output_size
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.cpln = CPLN(cfg)
        self.cls_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )
        
    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        features_list = [features[f] for f in self.in_features]
        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        loss_margin = None
        if self.training:
            _, rec_features, loss_margin = self.cpln.loss(box_features, proposals)
        else:
            rec_features = self.cpln.inference(box_features)
        pred_class_logits, cls_x, _ = self.cls_predictor(rec_features)
        _, _, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features
        
        outputs = ECFSODOutputs(
            cls_x,
            self.input_size,
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            loss = outputs.losses()
            loss["loss_margin"] = loss_margin
            return loss
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances