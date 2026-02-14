"""
CDAN: Conditional Domain Adversarial Network

Paper: "Conditional Adversarial Domain Adaptation" (NeurIPS 2018)
Authors: Long et al.

Implementation: Uses TLL (Transfer-Learning-Library) official components:
  - ConditionalDomainAdversarialLoss: conditions on classifier predictions
  - RandomizedMultiLinearMap: efficient multilinear conditioning
  - DomainDiscriminator: 3-layer MLP with BN

Source: tllib/alignment/cdan.py
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DAMethod
from ._tll_imports import (
    WarmStartGradientReverseLayer,
    DomainDiscriminator,
    ConditionalDomainAdversarialLoss,
)


class CDAN(DAMethod):
    """
    Conditional Domain Adversarial Network (CDAN).

    Uses TLL's official ConditionalDomainAdversarialLoss which handles:
    - Multilinear conditioning of features on classifier predictions
    - Gradient reversal with warm-start alpha annealing
    - Optional entropy conditioning
    - Optional randomized multilinear map for efficiency
    """

    NAME = "cdan"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        hidden_size: int = 1024,
        bottleneck_dim: int = 256,
        trade_off: float = 1.0,
        randomized: bool = True,
        randomized_dim: int = 1024,
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

        # Discriminator input dimension depends on conditioning
        if randomized:
            disc_input_dim = randomized_dim
        else:
            disc_input_dim = bottleneck_dim * num_classes

        # TLL Domain Discriminator (official architecture)
        self.domain_discriminator = DomainDiscriminator(
            in_feature=disc_input_dim,
            hidden_size=hidden_size,
            batch_norm=True,
            sigmoid=True,
        ).to(device)

        # TLL ConditionalDomainAdversarialLoss (official)
        # Handles: multilinear conditioning + GRL + discriminator + loss
        self.domain_adv_loss = ConditionalDomainAdversarialLoss(
            domain_discriminator=self.domain_discriminator,
            entropy_conditioning=False,
            randomized=randomized,
            num_classes=num_classes,
            features_dim=bottleneck_dim,
            randomized_dim=randomized_dim,
            reduction='mean',
            sigmoid=True,
        ).to(device)

        self.feature_dim = bottleneck_dim

    def observe(
        self,
        x_source: torch.Tensor,
        y_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Dict[str, float]:
        """Process a training batch with conditional domain adaptation."""
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

        # Classification
        y_pred_source = self.classifier(b_source)
        y_pred_target = self.classifier(b_target)

        cls_loss = F.cross_entropy(y_pred_source, y_source)

        # Conditional domain adversarial loss (TLL official)
        # ConditionalDomainAdversarialLoss.forward(g_s, f_s, g_t, f_t)
        # g = classifier predictions (softmax applied internally)
        # f = bottleneck features
        domain_loss = self.domain_adv_loss(
            y_pred_source, b_source, y_pred_target, b_target
        )

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
        params = (
            list(self.backbone.parameters()) +
            list(self.bottleneck.parameters()) +
            list(self.classifier.parameters()) +
            list(self.domain_discriminator.parameters())
        )
        return params

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
