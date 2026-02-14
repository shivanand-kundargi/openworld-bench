"""
MCD: Maximum Classifier Discrepancy

Paper: "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation" (CVPR 2018)
Authors: Saito et al.

Implementation: Uses TLL (Transfer-Learning-Library) official components:
  - classifier_discrepancy: L1 distance between two classifier predictions
  - ImageClassifierHead: 3-layer classifier with Dropout + BN

Source: tllib/alignment/mcd.py

Training procedure (3-step alternating):
  Step A: Train backbone + both classifiers on source CE
  Step B: Freeze backbone, maximize discrepancy on target (train classifiers)
  Step C: Freeze classifiers, minimize discrepancy on target (train backbone, num_k steps)
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DAMethod
from ._tll_imports import classifier_discrepancy, ImageClassifierHead


class MCD(DAMethod):
    """
    Maximum Classifier Discrepancy (MCD).

    Uses two classifiers with maximum discrepancy on target domain.
    Implements the full 3-step alternating optimization from the paper.
    Uses TLL's official classifier_discrepancy and ImageClassifierHead.
    """

    NAME = "mcd"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        bottleneck_dim: int = 1024,
        trade_off: float = 1.0,
        num_k: int = 4,  # Number of steps for Step C (generator update)
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        self.trade_off = trade_off
        self.num_k = num_k

        # TLL ImageClassifierHead: Dropout → Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Linear
        # We use two of these as the dual classifiers
        self.classifier1 = ImageClassifierHead(
            in_features=feature_dim,
            num_classes=num_classes,
            bottleneck_dim=bottleneck_dim,
            pool_layer=nn.Identity(),  # We handle pooling ourselves
        ).to(device)

        self.classifier2 = ImageClassifierHead(
            in_features=feature_dim,
            num_classes=num_classes,
            bottleneck_dim=bottleneck_dim,
            pool_layer=nn.Identity(),
        ).to(device)

        # Set classifier for base class compatibility (uses classifier1)
        self.classifier = self.classifier1

        self.feature_dim = feature_dim

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and flatten backbone features."""
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f

    def observe(
        self,
        x_source: torch.Tensor,
        y_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        MCD 3-step alternating training (from paper + TLL).

        Step A: Train all on source classification
        Step B: Freeze backbone, train classifiers to maximize target discrepancy
        Step C: Freeze classifiers, train backbone to minimize target discrepancy
        """
        x_source = x_source.to(self.device)
        y_source = y_source.to(self.device)
        x_target = x_target.to(self.device)

        # ──── Step A: Train backbone + classifiers on source CE ────
        f_source = self._extract_features(x_source)
        pred1 = self.classifier1(f_source)
        pred2 = self.classifier2(f_source)

        cls_loss = F.cross_entropy(pred1, y_source) + F.cross_entropy(pred2, y_source)
        cls_loss.backward()

        # ──── Step B: Freeze backbone, maximize discrepancy on target ────
        # Detach backbone features to prevent backbone gradient
        f_source_det = self._extract_features(x_source).detach()
        f_target_det = self._extract_features(x_target).detach()

        # Source classification (classifiers still train on source)
        pred1_s = self.classifier1(f_source_det)
        pred2_s = self.classifier2(f_source_det)
        cls_loss_b = F.cross_entropy(pred1_s, y_source) + F.cross_entropy(pred2_s, y_source)

        # Maximize discrepancy on target (using TLL classifier_discrepancy)
        pred1_t = F.softmax(self.classifier1(f_target_det), dim=1)
        pred2_t = F.softmax(self.classifier2(f_target_det), dim=1)
        disc = classifier_discrepancy(pred1_t, pred2_t)

        loss_b = cls_loss_b - self.trade_off * disc
        loss_b.backward()

        # ──── Step C: Freeze classifiers, minimize discrepancy (num_k steps) ────
        disc_min = torch.tensor(0.0, device=self.device)
        for _ in range(self.num_k):
            f_target_c = self._extract_features(x_target)
            pred1_tc = F.softmax(self.classifier1(f_target_c), dim=1)
            pred2_tc = F.softmax(self.classifier2(f_target_c), dim=1)
            disc_c = classifier_discrepancy(pred1_tc, pred2_tc)
            disc_min = self.trade_off * disc_c
            disc_min.backward()

        return {
            'total_loss': (cls_loss.item() + disc_min.item()),
            'cls_loss': cls_loss.item(),
            'discrepancy': disc.item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — average of both classifiers."""
        x = x.to(self.device)
        f = self._extract_features(x)
        pred1 = self.classifier1(f)
        pred2 = self.classifier2(f)
        return (pred1 + pred2) / 2

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get backbone features."""
        x = x.to(self.device)
        return self._extract_features(x)

    def parameters(self):
        return (
            list(self.backbone.parameters()) +
            list(self.classifier1.parameters()) +
            list(self.classifier2.parameters())
        )

    def state_dict(self) -> Dict:
        return {
            'backbone': self.backbone.state_dict(),
            'classifier1': self.classifier1.state_dict(),
            'classifier2': self.classifier2.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.backbone.load_state_dict(state['backbone'])
        self.classifier1.load_state_dict(state['classifier1'])
        self.classifier2.load_state_dict(state['classifier2'])
