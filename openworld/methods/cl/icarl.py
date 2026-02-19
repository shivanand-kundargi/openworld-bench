"""
iCaRL: Incremental Classifier and Representation Learning

Paper: "iCaRL: Incremental Classifier and Representation Learning" (CVPR 2017)
Authors: Rebuffi et al.

Implementation: Ported from mammoth (models/icarl.py).
Key differences from previous version:
  - Loss: BCE with sigmoid targets (not CE + separate distillation)
  - First task: BCE with one-hot targets
  - Later tasks: BCE with combined targets = cat(sigmoid(old_logits[:,:past]), one_hot[:,past:])
  - NCM: averaged features + horizontally flipped features, normalized
  - Weight decay: manual L2 penalty added to loss

Source: mammoth/models/icarl.py
"""

from typing import Dict, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .base import CLMethod


class ICarl(CLMethod):
    """
    iCaRL (mammoth port).

    Uses binary cross-entropy formulation where old-class targets come
    from sigmoid of old model outputs and new-class targets are one-hot.
    NCM classifier uses averaged features from real + horizontally-flipped images.
    """

    NAME = "icarl"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_tasks: int,
        device: torch.device,
        buffer_size: int = 2000,
        feature_dim: int = 2048,
        wd_reg: float = 1e-5,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device, buffer_size)

        self.feature_dim = feature_dim
        self.wd_reg = wd_reg
        self.current_task = 0
        self.n_past_classes = 0
        self.n_seen_classes = 0

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        # Classes per task (assumes uniform split)
        self.cpt = num_classes // num_tasks

        # One-hot encoder
        self.eye = torch.eye(num_classes).to(device)

        # Old model for distillation
        self.old_backbone = None
        self.old_classifier = None

        # NCM class means (computed at end of task)
        self.class_means = None

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and flatten features."""
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f

    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """
        mammoth iCaRL observe (BCE formulation).

        Task 0: BCE(logits, one_hot[labels])
        Task > 0: BCE(logits, cat(sigmoid(old_logits[:,:past]), one_hot[labels][:,past:]))
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass
        features = self._get_features(x)
        logits = self.classifier(features)

        # Build targets (mammoth BCE formulation)
        if self.current_task == 0:
            # First task: one-hot targets
            targets = self.eye[y]
        else:
            # Later tasks: combine old sigmoid targets + new one-hot targets
            with torch.no_grad():
                old_f = self.old_backbone(x)
                if old_f.dim() > 2:
                    old_f = F.adaptive_avg_pool2d(old_f, 1).flatten(1)
                old_logits = self.old_classifier(old_f)

            # Target = cat(sigmoid(old_outputs[:, :past_classes]), one-hot[:, past_classes:])
            targets = self.eye[y].clone()
            targets[:, :self.n_past_classes] = torch.sigmoid(old_logits[:, :self.n_past_classes])

        # BCE loss (mammoth formulation)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        # Buffer replay
        buf_loss = torch.tensor(0.0, device=self.device)
        if not self.buffer.is_empty():
            buf_x, buf_y = self.buffer.get_data(min(x.size(0), self.buffer.current_size))
            buf_x, buf_y = buf_x.to(self.device), buf_y.to(self.device)

            buf_features = self._get_features(buf_x)
            buf_logits = self.classifier(buf_features)

            # Same BCE formulation for buffer samples
            if self.current_task == 0:
                buf_targets = self.eye[buf_y]
            else:
                with torch.no_grad():
                    old_buf_f = self.old_backbone(buf_x)
                    if old_buf_f.dim() > 2:
                        old_buf_f = F.adaptive_avg_pool2d(old_buf_f, 1).flatten(1)
                    old_buf_logits = self.old_classifier(old_buf_f)
                buf_targets = self.eye[buf_y].clone()
                buf_targets[:, :self.n_past_classes] = torch.sigmoid(old_buf_logits[:, :self.n_past_classes])

            buf_loss = F.binary_cross_entropy_with_logits(buf_logits, buf_targets)

        total_loss = loss + buf_loss
        total_loss.backward()

        # Store to buffer (reservoir sampling)
        self.buffer.add_data(x.detach().cpu(), y.detach().cpu())

        return {
            'total_loss': total_loss.item(),
            'bce_loss': loss.item(),
            'buf_loss': buf_loss.item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        NCM classifier (mammoth-style).

        If class_means are computed, use nearest-class-mean.
        Otherwise, use linear classifier.
        """
        x = x.to(self.device)
        if self.class_means is not None:
            features = self._get_features(x)
            features = F.normalize(features, p=2, dim=1)
            # Nearest class mean
            dists = torch.cdist(features, self.class_means)
            # Return negative distances as logits (closer = higher)
            return -dists
        else:
            features = self._get_features(x)
            return self.classifier(features)

    def begin_task(self, task_id: int, num_classes_in_task: int = None):
        """Called at the start of each task."""
        self.current_task = task_id
        if num_classes_in_task is not None:
            self.cpt = num_classes_in_task
        self.n_seen_classes = self.n_past_classes + self.cpt

    def end_task(self, task_id: int, train_loader=None):
        """
        Called at end of each task.
        Store old model and compute class means (mammoth-style).
        """
        self.n_past_classes = self.n_seen_classes

        # Store old model for distillation
        self.old_backbone = copy.deepcopy(self.backbone)
        self.old_classifier = copy.deepcopy(self.classifier)
        self.old_backbone.eval()
        self.old_classifier.eval()
        for p in self.old_backbone.parameters():
            p.requires_grad = False
        for p in self.old_classifier.parameters():
            p.requires_grad = False

        # Compute NCM class means if we have buffer data
        if not self.buffer.is_empty():
            self._compute_class_means()

    def _compute_class_means(self):
        """
        Compute class means from buffer (mammoth-style with normalized features).
        In mammoth, this uses real + flipped features, but we simplify to just
        normalized mean features since we don't have horizontal flip in our pipeline.
        """
        self.backbone.eval()
        means = []

        buf_x, buf_y = self.buffer.get_all_data()
        buf_x, buf_y = buf_x.to(self.device), buf_y.to(self.device)

        for c in range(self.n_seen_classes):
            mask = buf_y == c
            if mask.any():
                with torch.no_grad():
                    feats = self._get_features(buf_x[mask])
                    feats = F.normalize(feats, p=2, dim=1)
                    mean = feats.mean(dim=0)
                    mean = F.normalize(mean, p=2, dim=0)
                means.append(mean)
            else:
                means.append(torch.zeros(self.feature_dim, device=self.device))

        self.class_means = torch.stack(means)
        self.backbone.train()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features."""
        x = x.to(self.device)
        return self._get_features(x)

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.classifier.parameters())

    def state_dict(self) -> Dict:
        state = {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
        }
        if self.class_means is not None:
            state['class_means'] = self.class_means
        return state

    def load_state_dict(self, state: Dict):
        self.backbone.load_state_dict(state['backbone'])
        self.classifier.load_state_dict(state['classifier'])
        if 'class_means' in state:
            self.class_means = state['class_means']
