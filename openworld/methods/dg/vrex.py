"""
VREx: Variance Risk Extrapolation

Paper: "Out-of-Distribution Generalization via Risk Extrapolation" (ICML 2021)
Authors: Krueger et al.

Implementation: Ported from DomainBed (facebookresearch/DomainBed).
Key components:
  - Variance penalty: ((losses - mean)^2).mean()
  - Penalty annealing: weight=1.0 before anneal_iters, then vrex_lambda
  - Concatenated forward pass, per-domain loss computation

Source: https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DGMethod


class VREx(DGMethod):
    """
    Variance Risk Extrapolation (DomainBed port).

    Penalizes the variance of per-domain risks to encourage learning
    features that have similar predictive quality across all domains.
    """

    NAME = "vrex"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        vrex_lambda: float = 1.0,
        vrex_penalty_anneal_iters: int = 500,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        self.vrex_lambda = vrex_lambda
        self.vrex_penalty_anneal_iters = vrex_penalty_anneal_iters
        self.update_count = 0

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        self.feature_dim = feature_dim

    def observe(
        self,
        x_domains: List[torch.Tensor],
        y_domains: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        DomainBed VREx update.

        loss = mean(losses) + penalty_weight * ((losses - mean)^2).mean()
        """
        self.update_count += 1

        # Penalty weight (DomainBed annealing)
        if self.update_count >= self.vrex_penalty_anneal_iters:
            penalty_weight = self.vrex_lambda
        else:
            penalty_weight = 1.0

        # DomainBed: concatenated forward pass
        all_x = torch.cat([x.to(self.device) for x in x_domains])
        all_f = self.backbone(all_x)
        if all_f.dim() > 2:
            all_f = F.adaptive_avg_pool2d(all_f, 1).flatten(1)
        all_logits = self.classifier(all_f)

        # Per-domain losses (DomainBed)
        losses = torch.zeros(len(x_domains))
        idx = 0
        nll = 0.0
        for i, (x, y) in enumerate(zip(x_domains, y_domains)):
            y = y.to(self.device)
            logits = all_logits[idx:idx + x.shape[0]]
            idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        # DomainBed VREx penalty: ((losses - mean)^2).mean()
        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()

        loss = mean + penalty_weight * penalty

        # Backward pass
        loss.backward()

        return {
            'total_loss': loss.item(),
            'nll': nll.item(),
            'penalty': penalty.item(),
            'penalty_weight': penalty_weight,
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
