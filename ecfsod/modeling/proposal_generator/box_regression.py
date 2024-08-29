import math
import torch
from typing import Tuple
from torch.nn import functional as F

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

__all__ = ["Box2BoxTransform", "Box2BoxTransformLinear"]

@torch.jit.script
class Box2BoxTransform(object):
    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights
        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights
        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        deltas = deltas.float()
        boxes = boxes.to(deltas.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes

class Box2BoxTransformLinear(object):
    def __init__(self, normalize_by_size=True):
        self.normalize_by_size = normalize_by_size

    def get_deltas(self, src_boxes, target_boxes):
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
        src_ctr_x = 0.5 * (src_boxes[:, 0] + src_boxes[:, 2])
        src_ctr_y = 0.5 * (src_boxes[:, 1] + src_boxes[:, 3])
        target_l = src_ctr_x - target_boxes[:, 0]
        target_t = src_ctr_y - target_boxes[:, 1]
        target_r = target_boxes[:, 2] - src_ctr_x
        target_b = target_boxes[:, 3] - src_ctr_y
        deltas = torch.stack((target_l, target_t, target_r, target_b), dim=1)
        if self.normalize_by_size:
            stride_w = src_boxes[:, 2] - src_boxes[:, 0]
            stride_h = src_boxes[:, 3] - src_boxes[:, 1]
            strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
            deltas = deltas / strides
        return deltas

    def apply_deltas(self, deltas, boxes):
        deltas = F.relu(deltas)
        boxes = boxes.to(deltas.dtype)
        ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        if self.normalize_by_size:
            stride_w = boxes[:, 2] - boxes[:, 0]
            stride_h = boxes[:, 3] - boxes[:, 1]
            strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
            deltas = deltas * strides
        l = deltas[:, 0::4]
        t = deltas[:, 1::4]
        r = deltas[:, 2::4]
        b = deltas[:, 3::4]
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = ctr_x[:, None] - l
        pred_boxes[:, 1::4] = ctr_y[:, None] - t
        pred_boxes[:, 2::4] = ctr_x[:, None] + r
        pred_boxes[:, 3::4] = ctr_y[:, None] + b
        return pred_boxes