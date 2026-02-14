"""
CORAL: CORrelation ALignment for Domain Generalization

Paper: "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (ECCV 2016)
Authors: Sun & Saenko

Implementation: Uses TLL (Transfer-Learning-Library) CorrelationAlignmentLoss
for pairwise domain alignment in a DG setting.

TLL's CORAL loss computes:
  loss = mean((C_s - C_t)^2) + mean((mu_s - mu_t)^2)
where C is the covariance matrix and mu is the mean per feature.

Source: tllib/alignment/coral.py
"""

import sys
import os
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DGMethod

# Import TLL CorrelationAlignmentLoss
_TLL_ROOT = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', 'Transfer-Learning-Library'
)
_TLL_ROOT = os.path.abspath(_TLL_ROOT)
if _TLL_ROOT not in sys.path:
    sys.path.insert(0, _TLL_ROOT)

from tllib.alignment.coral import CorrelationAlignmentLoss


class CORAL(DGMethod):
    """
    CORAL for Domain Generalization (TLL port).

    Minimizes domain shift by aligning second-order statistics (covariance)
    of features across all pairs of source domains using TLL's
    CorrelationAlignmentLoss.
    """

    NAME = "coral"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        penalty_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        self.penalty_weight = penalty_weight

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        # TLL CorrelationAlignmentLoss (official)
        self.coral_loss_fn = CorrelationAlignmentLoss()

        self.feature_dim = feature_dim

    def observe(
        self,
        x_domains: List[torch.Tensor],
        y_domains: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Process batches from multiple source domains.

        Loss = mean(CE per domain) + penalty_weight * mean(CORAL between all domain pairs)
        """
        # Collect per-domain features and losses
        domain_features = []
        cls_loss = 0.0

        for x, y in zip(x_domains, y_domains):
            x = x.to(self.device)
            y = y.to(self.device)

            f = self.backbone(x)
            if f.dim() > 2:
                f = F.adaptive_avg_pool2d(f, 1).flatten(1)

            logits = self.classifier(f)
            cls_loss += F.cross_entropy(logits, y)
            domain_features.append(f)

        cls_loss /= len(x_domains)

        # Pairwise CORAL alignment using TLL (official)
        coral_penalty = torch.tensor(0.0, device=self.device)
        n_pairs = 0
        for i in range(len(domain_features)):
            for j in range(i + 1, len(domain_features)):
                coral_penalty += self.coral_loss_fn(
                    domain_features[i], domain_features[j]
                )
                n_pairs += 1

        if n_pairs > 0:
            coral_penalty /= n_pairs

        total_loss = cls_loss + self.penalty_weight * coral_penalty

        # Backward pass
        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'coral_penalty': coral_penalty.item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features."""
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f
