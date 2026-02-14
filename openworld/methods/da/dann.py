"""
DANN: Domain-Adversarial Training of Neural Networks

Paper: "Domain-Adversarial Training of Neural Networks" (ICML 2015)
Authors: Yaroslav Ganin et al.

Implementation: Uses TLL (Transfer-Learning-Library) official components:
  - WarmStartGradientReverseLayer: alpha anneals from lo→hi over max_iters
  - DomainDiscriminator: 3-layer MLP with BN
  - DomainAdversarialLoss: wraps GRL + discriminator + BCE

Source: tllib/alignment/dann.py
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DAMethod
from ._tll_imports import (
    WarmStartGradientReverseLayer,
    GradientReverseFunction,
    DomainDiscriminator,
    DomainAdversarialLoss,
)


class GradientReversalLayer(nn.Module):
    """
    Simple GRL that holds a fixed coefficient (alpha).
    Needed by ToAlign and other methods that expect a stateful GRL.
    """
    def __init__(self, alpha: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)



class DANN(DAMethod):
    """
    Domain-Adversarial Neural Network (DANN).

    Uses TLL's official DomainAdversarialLoss which internally handles:
    - Gradient reversal with warm-start alpha annealing
    - Domain discrimination with binary cross-entropy
    - Separate source/target loss computation
    """

    NAME = "dann"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        hidden_size: int = 1024,
        bottleneck_dim: int = 256,
        trade_off: float = 1.0,
        max_iters: int = 1000,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        self.trade_off = trade_off

        # Bottleneck layer (same as TLL ImageClassifier)
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        ).to(device)

        # Task classifier
        self.classifier = nn.Linear(bottleneck_dim, num_classes).to(device)

        # TLL Domain Discriminator (official architecture)
        self.domain_discriminator = DomainDiscriminator(
            in_feature=bottleneck_dim,
            hidden_size=hidden_size,
            batch_norm=True,
            sigmoid=True,
        ).to(device)

        # TLL DomainAdversarialLoss with warm-start GRL
        # This internally creates WarmStartGradientReverseLayer
        self.domain_adv_loss = DomainAdversarialLoss(
            domain_discriminator=self.domain_discriminator,
            reduction='mean',
            grl=WarmStartGradientReverseLayer(
                alpha=1.0, lo=0.0, hi=1.0,
                max_iters=max_iters, auto_step=True,
            ),
            sigmoid=True,
        ).to(device)

        self.feature_dim = bottleneck_dim

    def observe(
        self,
        x_source: torch.Tensor,
        y_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Dict[str, float]:
        """Process a training batch with source and target domain data."""
        x_source = x_source.to(self.device)
        y_source = y_source.to(self.device)
        x_target = x_target.to(self.device)

        # Extract features
        f_source = self.backbone(x_source)
        f_target = self.backbone(x_target)

        if f_source.dim() > 2:
            f_source = F.adaptive_avg_pool2d(f_source, 1).flatten(1)
            f_target = F.adaptive_avg_pool2d(f_target, 1).flatten(1)

        # Bottleneck
        b_source = self.bottleneck(f_source)
        b_target = self.bottleneck(f_target)

        # Classification loss (source only)
        y_pred = self.classifier(b_source)
        cls_loss = F.cross_entropy(y_pred, y_source)

        # Domain adversarial loss (TLL official)
        # DomainAdversarialLoss.forward(f_s, f_t) handles GRL + discriminator + BCE
        domain_loss = self.domain_adv_loss(b_source, b_target)

        # Total loss
        total_loss = cls_loss + self.trade_off * domain_loss

        # Backward pass
        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'domain_loss': domain_loss.item(),
            'domain_acc': self.domain_adv_loss.domain_discriminator_accuracy,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        b = self.bottleneck(f)
        return self.classifier(b)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get bottleneck features."""
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.bottleneck(f)

    def parameters(self):
        """Get all trainable parameters."""
        return (
            list(self.backbone.parameters()) +
            list(self.bottleneck.parameters()) +
            list(self.classifier.parameters()) +
            list(self.domain_discriminator.parameters())
        )

    def state_dict(self) -> Dict:
        return {
            'backbone': self.backbone.state_dict(),
            'bottleneck': self.bottleneck.state_dict(),
            'classifier': self.classifier.state_dict(),
            'domain_discriminator': self.domain_discriminator.state_dict(),
        }

    def load_state_dict(self, state: Dict):
        self.backbone.load_state_dict(state['backbone'])
        self.bottleneck.load_state_dict(state['bottleneck'])
        self.classifier.load_state_dict(state['classifier'])
        self.domain_discriminator.load_state_dict(state['domain_discriminator'])
