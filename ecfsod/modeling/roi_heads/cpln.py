import torch
import logging
from torch import nn
from typing import List
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.layers import nonzero_tuple, cat

logger = logging.getLogger(__name__)

class CPLN(nn.Module):
    @configurable
    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 1024,
        embedding_dim: int = 256,
        distance_type: str = "COS",
        reps_per_class: int = 1,
        alpha: float = 0.2,
        beta: float = 0.8,
        loss_weight: float = 1.0,
        iou_threshold: float = 0.7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.distance_type = distance_type
        self.reps_per_class = reps_per_class
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight
        self.iou_threshold = iou_threshold

        self.encoder = nn.Linear(self.feature_dim, self.embedding_dim).to(torch.device('cuda'))
        nn.init.normal_(self.encoder.weight, std=0.01)
        nn.init.constant_(self.encoder.bias, 0)

        self.decoder = nn.Linear(self.embedding_dim, self.feature_dim).to(torch.device('cuda'))
        nn.init.normal_(self.decoder.weight, std=0.01)
        nn.init.constant_(self.decoder.bias, 0)

        self.representatives = nn.parameter.Parameter(
            torch.zeros(self.num_classes * self.reps_per_class, self.embedding_dim)
        )
        nn.init.normal_(self.representatives)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes"       : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "feature_dim"       : cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            "embedding_dim"     : cfg.MODEL.CPLN.EMD_DIM,
            "distance_type"     : cfg.MODEL.CPLN.DISTANCE_TYPE,
            "reps_per_class"    : cfg.MODEL.CPLN.REPS_PER_CLASS,
            "alpha"             : cfg.MODEL.CPLN.ALPHA,
            "beta"              : cfg.MODEL.CPLN.BETA,
            "loss_weight"       : cfg.MODEL.CPLN.LOSS_WEIGHT, 
            "iou_threshold"     : cfg.MODEL.CPLN.IOU_THRESHOLD,
        }

    def loss(self, roi_features: torch.Tensor, proposals: List[Instances]):
        emb_features = self.encoder(roi_features)
        new_features = F.normalize(emb_features)
        rec_features = self.decoder(emb_features)

        representatives = F.normalize(self.representatives) 
        ious = (cat([p.ious for p in proposals], dim=0) if len(proposals) else torch.empty(0))
        gt_classes = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes) & (ious > self.iou_threshold))[0]
        new_features = new_features[fg_inds]

        if self.distance_type == 'L1':
            dist = torch.cdist(new_features, representatives, p=1.0)
        elif self.distance_type == 'L2':
            dist = torch.cdist(new_features, representatives)
        elif self.distance_type == 'COS':
            dist = 1.0 - torch.mm(new_features, representatives.transpose(0,1))
        
        min_dist, _ = torch.min(dist.reshape(-1, self.num_classes, self.reps_per_class), dim=2)
        intra_dist = min_dist[torch.arange(min_dist.shape[0]), gt_classes[fg_inds]] 
        min_dist[torch.arange(min_dist.shape[0]), gt_classes[fg_inds]] = 1000
        inter_dist, _ = torch.min(min_dist, dim=1)

        if self.distance_type == 'L1':
            center_dist = torch.cdist(representatives, representatives, p=1.0)
        elif self.distance_type == 'L2':
            center_dist = torch.cdist(representatives, representatives)
        elif self.distance_type == 'COS':
            center_dist = 1.0 - torch.mm(representatives, representatives.transpose(0,1))

        center_dist_clone = center_dist.clone()
        for i in range(self.num_classes):
            center_dist_clone[i * self.reps_per_class:(i+1)*self.reps_per_class,  i * self.reps_per_class:(i+1)*self.reps_per_class] = 1000
        c_dist, _ = torch.min(center_dist_clone, dim=1)

        dml_loss = torch.sum(torch.max(intra_dist-self.alpha, torch.zeros_like(intra_dist))) + \
            torch.sum(torch.max(self.beta - inter_dist, torch.zeros_like(inter_dist))) + \
            torch.sum(torch.max(self.beta + self.alpha - c_dist, torch.zeros_like(c_dist)))
        
        return emb_features, rec_features, dml_loss * self.loss_weight / max(gt_classes.numel(), 1.0)

    def inference(self, roi_features: torch.Tensor):
        emb_features_per_image = self.encoder(roi_features)
        rec_features_per_image = self.decoder(emb_features_per_image)
        return rec_features_per_image