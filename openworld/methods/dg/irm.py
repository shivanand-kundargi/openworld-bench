"""
IRM: Invariant Risk Minimization

Paper: "Invariant Risk Minimization" (arXiv 2019)
Authors: Arjovsky et al.

Implementation: Ported from DomainBed (facebookresearch/DomainBed).
Key components:
  - IRM penalty: grad(loss_1 * scale, [scale]) * grad(loss_2 * scale, [scale])
    where loss is split into two halves for the penalty computation
  - Penalty annealing: weight=1.0 before anneal_iters, then irm_lambda
  - Per-domain loss computation with averaging

Source: https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from .base import DGMethod


class IRM(DGMethod):
    """
    Invariant Risk Minimization (DomainBed port).

    Penalizes variance of optimal classifiers across domains to
    encourage learning invariant representations.
    """

    NAME = "irm"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        irm_lambda: float = 1.0,
        irm_penalty_anneal_iters: int = 500,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        self.irm_lambda = irm_lambda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.update_count = 0

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        self.feature_dim = feature_dim

    @staticmethod
    def _irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        DomainBed IRM penalty.

        Splits batch into two halves, computes gradient of each half's loss
        w.r.t. a dummy scale factor, returns dot product of gradients.
        """
        device = logits.device
        scale = torch.tensor(1., device=device, requires_grad=True)
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def observe(
        self,
        x_domains: List[torch.Tensor],
        y_domains: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        DomainBed IRM update.

        loss = mean(nll_per_domain) + penalty_weight * mean(irm_penalty_per_domain)
        """
        self.update_count += 1

        # Penalty weight (DomainBed annealing)
        if self.update_count >= self.irm_penalty_anneal_iters:
            penalty_weight = self.irm_lambda
        else:
            penalty_weight = 1.0

        nll = 0.0
        penalty = 0.0

        # DomainBed: concatenated forward pass, then per-domain loss
        all_x = torch.cat([x.to(self.device) for x in x_domains])
        all_f = self.backbone(all_x)
        if all_f.dim() > 2:
            all_f = F.adaptive_avg_pool2d(all_f, 1).flatten(1)
        all_logits = self.classifier(all_f)

        idx = 0
        for x, y in zip(x_domains, y_domains):
            y = y.to(self.device)
            logits = all_logits[idx:idx + x.shape[0]]
            idx += x.shape[0]

            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)

        nll /= len(x_domains)
        penalty /= len(x_domains)

        loss = nll + (penalty_weight * penalty)

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
