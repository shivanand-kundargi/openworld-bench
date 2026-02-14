"""
MIRO: Mutual Information Regularization with Oracle

Paper: "Domain Generalization by Mutual-Information Regularization with
Pre-trained Models" (ECCV 2022)
Official source: https://github.com/kakaobrain/miro (khanrc/miro)

Key Idea: Regularize learned features to stay close to pre-trained
representations using a variational lower bound on mutual information.
Adapted from: khanrc/miro/domainbed/algorithms/miro.py
"""

from typing import Dict, List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DGMethod


class MeanEncoder(nn.Module):
    """
    Identity function for mean estimation.
    Adapted from khanrc/miro/domainbed/algorithms/miro.py::MeanEncoder.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """
    Bias-only model with diagonal covariance.
    Adapted from khanrc/miro/domainbed/algorithms/miro.py::VarianceEncoder.
    """

    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init_val = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape) == 3:
                # ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            elif len(shape) == 2:
                # [B, C] — standard feature vector
                b_shape = (1, shape[1])
            else:
                b_shape = shape

        self.b = nn.Parameter(torch.full(b_shape, init_val))

    def forward(self, x):
        return F.softplus(self.b) + self.eps


class MIRO(DGMethod):
    """
    MIRO: Mutual Information Regularization with Oracle.

    Uses a frozen pre-trained backbone as an "oracle" and regularizes
    the learning backbone's intermediate features to not deviate from
    the oracle via a variational lower bound on mutual information.

    Adapted from khanrc/miro/domainbed/algorithms/miro.py::MIRO.
    """

    NAME = "miro"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        ld: float = 0.1,             # MI regularization weight (lambda)
        lr_mult: float = 10.0,       # LR multiplier for mean/var encoders
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        self.ld = ld
        self.lr_mult = lr_mult

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        # Pre-trained oracle backbone — frozen copy
        self.pre_featurizer = deepcopy(backbone)
        for param in self.pre_featurizer.parameters():
            param.requires_grad = False
        self.pre_featurizer = self.pre_featurizer.to(device)
        self.pre_featurizer.eval()

        # Mean / Variance encoders for MI estimation
        # For standard backbones outputting [B, feature_dim], we use
        # a single pair of encoders on the final feature.
        feat_shape = (1, feature_dim)
        self.mean_encoder = MeanEncoder(feat_shape).to(device)
        self.var_encoder = VarianceEncoder(feat_shape).to(device)

        self.feature_dim = feature_dim

    def observe(
        self,
        x_domains: List[torch.Tensor],
        y_domains: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Process training batch with MIRO MI regularization.
        
        Core logic from khanrc/miro miro.py::MIRO.update():
          loss = CE(logit, y)
          reg  = 0.5 * mean( (mean - pre_f)^2 / var  +  log(var) )
          loss += ld * reg
        """
        all_x = torch.cat(x_domains).to(self.device)
        all_y = torch.cat(y_domains).to(self.device)

        # Student features
        feat = self.backbone(all_x)
        if feat.dim() > 2:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)

        # Oracle features (frozen pre-trained)
        with torch.no_grad():
            pre_feat = self.pre_featurizer(all_x)
            if pre_feat.dim() > 2:
                pre_feat = F.adaptive_avg_pool2d(pre_feat, 1).flatten(1)

        # MI regularization  (variational lower bound)
        # From official code:
        #   mean = mean_enc(f);  var = var_enc(f)
        #   vlb  = (mean - pre_f)^2 / var  +  log(var)
        #   reg  = vlb.mean() / 2
        mean = self.mean_encoder(feat)          # identity
        var = self.var_encoder(feat)             # softplus(b) + eps
        vlb = (mean - pre_feat).pow(2).div(var) + var.log()
        reg_loss = vlb.mean() / 2.0

        total_loss = loss + self.ld * reg_loss

        # Backward
        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'cls_loss': loss.item(),
            'reg_loss': reg_loss.item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f

    def parameters(self):
        """
        Return trainable parameters.
        Note: mean_encoder has no learnable params (identity);
              var_encoder and classifier are trainable.
        """
        return (
            list(self.backbone.parameters()) +
            list(self.classifier.parameters()) +
            list(self.var_encoder.parameters())
        )
