import torch
from typing import List, Union
from detectron2.layers import cat
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling.box_regression import Box2BoxTransform

def iou_loss(
    anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_mask: torch.Tensor,
    box_reg_loss_type="iou",
):
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor
    else:
        anchors = cat(anchors)
    if box_reg_loss_type == "iou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        pred_boxes = Boxes(torch.stack(pred_boxes)[fg_mask])
        gt_boxes = Boxes(torch.stack(gt_boxes)[fg_mask])
        ious = torch.diag(pairwise_iou(pred_boxes, gt_boxes)).clamp(min=1e-6)
        loss_box_reg = torch.sum(1 - ious)
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg