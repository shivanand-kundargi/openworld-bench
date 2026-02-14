"""
ToAlign: Task-oriented Alignment for Unsupervised Domain Adaptation

Paper: "ToAlign: Task-oriented Alignment for Unsupervised Domain Adaptation"
(NeurIPS 2021)
Official source: https://github.com/microsoft/UDA

Key Idea: Decompose features into task-relevant and task-irrelevant components
using classifier weights, then align only the task-relevant parts.
Adapted from: microsoft/UDA/trainer/da/toalign.py + models/base_model.py
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DAMethod
from .dann import GradientReversalLayer, DomainDiscriminator


class ToAlign(DAMethod):
    """
    ToAlign: Task-oriented Alignment.

    Core innovation from official code (microsoft/UDA/models/base_model.py):
      - Uses classifier weights to decompose features into task-relevant 
        and task-irrelevant components.
      - _get_toalign_weight(): extracts per-sample weights from the 
        classifier weight matrix indexed by class label, energy-normalised.
      - HDA (Hierarchical Domain Alignment): optional multi-head 
        decomposition (fc, fc0, fc1, fc2).

    We simplify HDA to single classifier head for our interface,
    keeping the core toalign weight decomposition faithful to official code.
    """

    NAME = "toalign"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        bottleneck_dim: int = 256,
        trade_off: float = 1.0,
        hda: bool = True,   # Enable HDA heads like official code
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        self.trade_off = trade_off
        self.hda = hda

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        ).to(device)

        # Main classifier
        self.classifier = nn.Linear(bottleneck_dim, num_classes).to(device)
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # HDA heads (from official base_model.py)
        if self.hda:
            self.fc0 = nn.Linear(bottleneck_dim, num_classes).to(device)
            self.fc1 = nn.Linear(bottleneck_dim, num_classes).to(device)
            self.fc2 = nn.Linear(bottleneck_dim, num_classes).to(device)
            # Different initialization for each (as in official code)
            nn.init.xavier_normal_(self.fc0.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.orthogonal_(self.fc2.weight)
            for fc in [self.fc0, self.fc1, self.fc2]:
                if fc.bias is not None:
                    nn.init.zeros_(fc.bias)

        # Domain discriminator (operates on classifier output dim)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_discriminator = DomainDiscriminator(
            in_features=num_classes,
            hidden_size=1024
        ).to(device)

        self.feature_dim = bottleneck_dim

    def _get_toalign_weight(self, f, labels):
        """
        Compute task-oriented feature weights.
        Faithfully adapted from microsoft/UDA/models/base_model.py::_get_toalign_weight.
        
        w = fc.weight[labels]  (per-sample classifier weight vector)
        If HDA: w = w - (w0 + w1 + w2)  (remove task-irrelevant directions)
        Then energy-normalise: scalar = sqrt(||f||^2 / ||f*w||^2)
        w_pos = w * scalar
        """
        w = self.classifier.weight[labels].detach()  # [B, bottleneck_dim]

        if self.hda:
            w0 = self.fc0.weight[labels].detach()
            w1 = self.fc1.weight[labels].detach()
            w2 = self.fc2.weight[labels].detach()
            w = w - (w0 + w1 + w2)

        eng_org = (f ** 2).sum(dim=1, keepdim=True)      # [B, 1]
        eng_aft = ((f * w) ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        scalar = (eng_org / (eng_aft + 1e-8)).sqrt()
        w_pos = w * scalar

        return w_pos

    def _forward_features(self, x, toalign=False, labels=None):
        """
        Forward pass returning (f, y, z) like official base_model.py.
        f = features, y = classification logits, z = HDA component.
        """
        raw_f = self.backbone(x)
        if raw_f.dim() > 2:
            raw_f = F.adaptive_avg_pool2d(raw_f, 1).flatten(1)
        f = self.bottleneck(raw_f)

        if toalign and labels is not None:
            w_pos = self._get_toalign_weight(f, labels)
            f_pos = f * w_pos
            y_pos = self.classifier(f_pos)
            if self.hda:
                z_pos = self.fc0(f_pos) + self.fc1(f_pos) + self.fc2(f_pos)
                return f_pos, y_pos - z_pos, z_pos
            else:
                return f_pos, y_pos, torch.zeros_like(y_pos)
        else:
            y = self.classifier(f)
            if self.hda:
                z = self.fc0(f) + self.fc1(f) + self.fc2(f)
                return f, y - z, z
            else:
                return f, y, torch.zeros_like(y)

    def observe(
        self,
        x_source: torch.Tensor,
        y_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Process training batch with task-oriented alignment.
        Adapted from microsoft/UDA/trainer/da/toalign.py::ToAlign.one_step.
        """
        x_source = x_source.to(self.device)
        y_source = y_source.to(self.device)
        x_target = x_target.to(self.device)

        # ---- Classification (source, no toalign) ----
        f_src, y_src, z_src = self._forward_features(x_source, toalign=False)
        loss_cls_src = F.cross_entropy(y_src, y_source)
        focals_src = z_src  # HDA focal term

        # ---- Alignment (source with toalign, target without) ----
        f_src_p, y_src_p, z_src_p = self._forward_features(
            x_source, toalign=True, labels=y_source
        )
        f_tgt, y_tgt, z_tgt = self._forward_features(x_target, toalign=False)
        focals_tgt = z_tgt

        # Concatenate logits for domain discrimination
        logits_all = torch.cat([y_src_p, y_tgt], dim=0)
        softmax_all = F.softmax(logits_all, dim=1)

        # Domain alignment via discriminator on softmax outputs
        reversed_softmax = self.grl(softmax_all)
        domain_pred = self.domain_discriminator(reversed_softmax)
        domain_labels = torch.cat([
            torch.ones(x_source.size(0), 1),
            torch.zeros(x_target.size(0), 1)
        ]).to(self.device)
        loss_alg = F.binary_cross_entropy(domain_pred, domain_labels)

        # HDA loss: keep focal terms small (from official: focals_all.abs().mean())
        focals_all = torch.cat([focals_src, focals_tgt], dim=0)
        loss_hda = focals_all.abs().mean()

        total_loss = loss_cls_src + self.trade_off * loss_alg + loss_hda

        # Backward
        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'cls_loss': loss_cls_src.item(),
            'domain_loss': loss_alg.item(),
            'hda_loss': loss_hda.item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        _, y, _ = self._forward_features(x, toalign=False)
        return y

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f, _, _ = self._forward_features(x, toalign=False)
        return f

    def parameters(self):
        params = (
            list(self.backbone.parameters()) +
            list(self.bottleneck.parameters()) +
            list(self.classifier.parameters()) +
            list(self.domain_discriminator.parameters())
        )
        if self.hda:
            params += (
                list(self.fc0.parameters()) +
                list(self.fc1.parameters()) +
                list(self.fc2.parameters())
            )
        return params
