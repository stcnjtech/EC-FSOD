import torch
from torch import nn
from .iou_loss import iou_loss
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from .build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.matcher import Matcher
from .box_regression import Box2BoxTransformLinear
from detectron2.layers import Conv2d, ShapeSpec, cat
from typing import Dict, List, Optional, Tuple, Union
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling import build_anchor_generator
from .find_top_proposals import find_top_rpn_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")

def torch_cov(input_vec:torch.tensor):    
    x = input_vec- torch.mean(input_vec,axis=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
    return cov_matrix

def build_rpn_head(cfg, input_shape):
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)

@RPN_HEAD_REGISTRY.register()
class ECFRPNHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        k_num: int,
        in_channels: int,
        num_anchors: int,
        box_dim: int = 4,
        conv_dims: List[int] = (-1,)
    ):
        super().__init__()
        cur_channels = in_channels
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)
        self.k = k_num
        self.centerness_list = []
        for _ in range(self.k):
            centerness = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
            nn.init.normal_(centerness.weight, std=0.01)
            nn.init.constant_(centerness.bias, 0)
            self.centerness_list.append(centerness.to("cuda"))
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {
            "k_num"         : cfg.MODEL.RPN.K_NUM,
            "in_channels"   : in_channels,
            "num_anchors"   : num_anchors[0],
            "box_dim"       : box_dim,
            "conv_dims"     : (-1,)
        }
    
    def forward(self, features: List[torch.Tensor]):
        pred_anchor_deltas = []
        pred_centerness = [[] for _ in range(len(features))]
        for idx, x in enumerate(features):
            t = self.conv(x)
            t = F.normalize(t, p=2, dim=1)
            pred_anchor_deltas.append(self.anchor_deltas(t))
            for i in range(self.k):
                pred_centerness[idx].append(self.centerness_list[i](t).sigmoid())
        return pred_anchor_deltas, pred_centerness

@PROPOSAL_GENERATOR_REGISTRY.register()
class ECFRPN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        objectness_anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransformLinear,
        batch_size_per_image: int,
        positive_fraction: float,
        objectness_positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: Tuple[float, float],
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        reg_loss_type: str = "smooth_l1",
        reg_smooth_l1_beta: float = 0.0,
        loc_loss_type: str = "smooth_l1",
        loc_smooth_l1_beta: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.objectness_anchor_matcher = objectness_anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.objectness_positive_fraction = objectness_positive_fraction
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = {True: nms_thresh[0], False: nms_thresh[1]}
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_reg": loss_weight, 
                           "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.reg_loss_type = reg_loss_type
        self.reg_smooth_l1_beta = reg_smooth_l1_beta
        self.loc_loss_type = loc_loss_type
        self.loc_smooth_l1_beta = loc_smooth_l1_beta

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES 
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "objectness_positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION_OBJECTNESS,
            "loss_weight": {
                "loss_rpn_reg": cfg.MODEL.RPN.REG_LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.LOC_LOSS_WEIGHT 
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransformLinear(normalize_by_size=True),
            "reg_loss_type": cfg.MODEL.RPN.REG_LOSS_TYPE,
            "reg_smooth_l1_beta": cfg.MODEL.RPN.REG_SMOOTH_L1_BETA,
            "loc_loss_type": cfg.MODEL.RPN.LOC_LOSS_TYPE,
            "loc_smooth_l1_beta": cfg.MODEL.RPN.LOC_SMOOTH_L1_BETA,
        }
        ret["nms_thresh"] = (cfg.MODEL.RPN.NMS_THRESH, cfg.MODEL.RPN.NMS_THRESH_TEST)
        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)
        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["objectness_anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS_OBJECTNESS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def _subsample_labels(self, label, pos_frac):
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, pos_frac, 0
        )
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        anchors = Boxes.cat(anchors)
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances
        gt_labels = []
        matched_gt_boxes = []
        objectness_gt_labels = []
        gt_centerness = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            objectness_matched_idxs, objectness_gt_labels_i = retry_if_cuda_oom(self.objectness_anchor_matcher)(match_quality_matrix)
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            objectness_gt_labels_i = objectness_gt_labels_i.to(device=gt_boxes_i.device)
            if self.anchor_boundary_thresh >= 0:
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1
                objectness_gt_labels_i[~anchors_inside_image] = -1
            gt_labels_i = self._subsample_labels(gt_labels_i, self.positive_fraction)
            objectness_gt_labels_i = self._subsample_labels(objectness_gt_labels_i, self.objectness_positive_fraction)
            if len(gt_boxes_i) == 0:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor
            matched_pairwise_dist = self.box2box_transform.get_deltas(anchors.tensor, gt_boxes_i.tensor[objectness_matched_idxs])
            matched_pairwise_dist = matched_pairwise_dist[:, [0,2,1,3]]
            is_in_boxes = (matched_pairwise_dist >= 0).all(dim=1)
            matched_pairwise_dist[~is_in_boxes, :] = 0
            left_right = matched_pairwise_dist[:,0:2]
            top_bottom = matched_pairwise_dist[:,2:4]
            gt_centerness_i = torch.sqrt(
                (torch.min(left_right, -1)[0] / (torch.max(left_right, -1)[0] + 1e-12)) * 
                (torch.min(top_bottom, -1)[0] / (torch.max(top_bottom, -1)[0] + 1e-12)))
            gt_centerness_i[objectness_gt_labels_i == 0] = 0.0
            del match_quality_matrix
            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            objectness_gt_labels.append(objectness_gt_labels_i)
            gt_centerness.append(gt_centerness_i)
        return gt_labels, matched_gt_boxes, objectness_gt_labels, gt_centerness

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        pred_centerness: List[torch.Tensor],
        gt_centerness: List[torch.Tensor],
        objectness_gt_labels: List[torch.Tensor],
        dis_loss: torch.Tensor = None,
        coop_loss: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)
        gt_centerness = torch.stack(gt_centerness)
        objectness_gt_labels = torch.stack(objectness_gt_labels)
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        obj_pos_mask = objectness_gt_labels == 1
        obj_mask = objectness_gt_labels != -1
        obj_num_pos_anchors = obj_pos_mask.sum().item()
        obj_num_neg_anchors = (objectness_gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)
        storage.put_scalar("rpn/obj_num_pos_anchors", obj_num_pos_anchors / num_images)
        storage.put_scalar("rpn/obj_num_neg_anchors", obj_num_neg_anchors / num_images)

        reg_loss = iou_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            self.reg_loss_type,
        )

        if self.loc_loss_type == "smooth_l1":
            loc_loss = smooth_l1_loss(
                cat(pred_centerness, dim=1)[obj_mask],
                gt_centerness[obj_mask],
                beta=self.loc_smooth_l1_beta,
                reduction="sum"
            )

        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_reg": reg_loss / normalizer,
            "loss_rpn_loc": loc_loss / normalizer,
            "dis_loss": dis_loss / normalizer,
            "coop_loss": coop_loss / normalizer
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_anchor_deltas, pred_centerness = self.rpn_head(features)
        
        pred_anchor_deltas = [
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        
        pred_centerness = [
            [   
                centerness.permute(0, 2, 3, 1).flatten(1)
                for centerness in feat_map
            ]
            for feat_map in pred_centerness
        ]
        
        pred_centerness_tensor = []
        for feat_map in pred_centerness:
            feat_tensor = torch.stack(feat_map)
            pred_centerness_tensor.append(feat_tensor)

        
        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes, objectness_gt_labels, gt_centerness = self.label_and_sample_anchors(anchors, gt_instances)
            
            target_pred_centerness = []
            for feat_tensor in pred_centerness_tensor:
                min_feat_tensor = torch.minimum(feat_tensor, 1 - feat_tensor)
                idx_min = torch.argmin(min_feat_tensor,dim=0,keepdim=True)
                min_pred_centerness_tensor = torch.gather(feat_tensor,0,idx_min).squeeze(0)
                target_pred_centerness.append(min_pred_centerness_tensor)

            dis_loss, coop_loss = None, None
            count = 0
            for feat_tensor in pred_centerness_tensor:
                coop_loss_1 = torch.max(torch.zeros_like(feat_tensor),0.5-feat_tensor)
                coop_loss_1 = torch.sum(coop_loss_1)
                cov_mat = torch_cov(feat_tensor.view(feat_tensor.shape[0],-1).T)
                dis_loss_1 = - torch.logdet(cov_mat.cpu())
                dis_loss_1 = dis_loss_1.to("cuda")
                count += 1
                if count == 1:
                    dis_loss = dis_loss_1
                    coop_loss = coop_loss_1
                else:
                    dis_loss += dis_loss_1
                    coop_loss += coop_loss_1

            dis_loss = dis_loss / count
            coop_loss = coop_loss / count
            
            losses = self.losses(
                anchors, gt_labels, 
                pred_anchor_deltas, gt_boxes,
                target_pred_centerness, gt_centerness,
                objectness_gt_labels,
                dis_loss,
                coop_loss,
            )
        else:
            losses = {}

        proposal_pred_centerness = []
        for feat_tensor in pred_centerness_tensor:
            idx_max = torch.argmax(feat_tensor,dim=0,keepdim=True)
            max_pred_centerness_tensor = torch.gather(feat_tensor,0,idx_max).squeeze(0)
            proposal_pred_centerness.append(max_pred_centerness_tensor)
        
        proposals = self.predict_proposals(
            anchors, pred_anchor_deltas, proposal_pred_centerness, images.image_sizes
        )
        
        num_proposals = 0
        for proposals_i in proposals:
            num_proposals += len(proposals_i)
        if self.training:
            storage = get_event_storage()
            storage.put_scalar("rpn/num_proposals", num_proposals / len(proposals))

        return proposals, losses

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_anchor_deltas: List[torch.Tensor],
        pred_centerness: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_centerness,
                image_sizes,
                self.nms_thresh[self.training],
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals