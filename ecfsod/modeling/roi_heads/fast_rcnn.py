import torch
import logging
import numpy as np
from torch import nn
from fvcore.nn import smooth_l1_loss
from torch.nn import functional as F
from detectron2.utils.registry import Registry
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

ROI_HEADS_OUTPUT_REGISTRY = Registry("ROI_HEADS_OUTPUT")

logger = logging.getLogger(__name__)

def fast_rcnn_inference(
    boxes, scores, image_shapes, 
    score_thresh, nms_thresh, topk_per_image
):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return tuple(list(x) for x in zip(*result_per_image))

def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, 
    score_thresh, nms_thresh, topk_per_image
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

class ECFSODOutputs(object):    
    def __init__(
        self,
        cls_x,
        input_size,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    ):
        self.cls_x = cls_x
        self.input_size = input_size
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        box_type = type(proposals[0].proposal_boxes)
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        num_class_prac = self.pred_class_logits.shape[1]
        weight = torch.sqrt(torch.tensor(num_class_prac/(num_class_prac-1)))*(torch.eye(num_class_prac)-(1/num_class_prac)*torch.ones((num_class_prac, num_class_prac)))
        weight /= torch.sqrt((1/num_class_prac*torch.norm(weight, 'fro')**2))
        self.w = torch.mm(weight, torch.eye(num_class_prac, self.input_size)).to("cuda")

    def _log_accuracy(self):
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]
        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def softmax_cross_entropy_loss(self):
        self._log_accuracy()
        return F.cross_entropy(
            self.pred_class_logits, self.gt_classes, reduction="mean"
        )

    def dot_regression_loss(self):
        x = F.normalize(self.cls_x, p=2, dim=1)
        dot = torch.sum(x * self.w[self.gt_classes,:], dim=1)
        return 0.5 * torch.mean((torch.ones_like(dot) - dot)**2)

    def smooth_l1_loss(self):
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        ).squeeze(1)
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )
        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        return {
            "loss_cls_ce": self.softmax_cross_entropy_loss(),
            "loss_cls_dr": self.dot_regression_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1)
            .expand(num_pred, K, B)
            .reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(
            self.num_preds_per_image, dim=0
        )

    def predict_probs(self):
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )

@ROI_HEADS_OUTPUT_REGISTRY.register()
class ECFSODOutputETFLayers(nn.Module):
    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg = False, box_dim=4):
        super(ECFSODOutputETFLayers, self).__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        self.mapping = nn.Linear(input_size, input_size,  bias=False)
        nn.init.normal_(self.mapping.weight, std=0.01)
        num_class_prac = num_classes + 1
        self.cls_score = nn.Linear(input_size, num_class_prac, bias=False)
        weight = torch.sqrt(torch.tensor(num_class_prac/(num_class_prac-1)))*(torch.eye(num_class_prac)-(1/num_class_prac)*torch.ones((num_class_prac, num_class_prac)))
        weight /= torch.sqrt((1/num_class_prac*torch.norm(weight, 'fro')**2))
        self.cls_score.weight = nn.Parameter(torch.mm(weight, torch.eye(num_class_prac, input_size)))
        self.cls_score.requires_grad_(False)
        self.etf_residual = cfg.MODEL.SEC.ETF_RESIDUAL
        logger.info("ETF Residusl {}".format(self.etf_residual))
        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        if self._do_cls_dropout:
            x = F.dropout(x, self._dropout_ratio, training=self.training)
        cls_x = self.mapping(x) + x if self.etf_residual else self.mapping(x)
        scores = self.cls_score(cls_x)
        return scores, cls_x, proposal_deltas