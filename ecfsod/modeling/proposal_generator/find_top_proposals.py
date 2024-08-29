import torch
import logging
from typing import List, Tuple
from detectron2.layers import cat
from detectron2.structures import Boxes, Instances

logger = logging.getLogger(__name__)

def _is_tracing():
    if torch.jit.is_scripting():
        return False
    else:
        return torch.jit.is_tracing()

def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    num_images = len(image_sizes)
    device = proposals[0].device
    topk_scores = []
    topk_proposals = []
    level_ids = []
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)
        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]
        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids
        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]
        res = Instances(image_size)
        res.proposal_boxes = boxes
        res.objectness_logits = scores_per_img
        results.append(res)
    return results