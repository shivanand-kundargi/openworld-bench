"""
SWAD: Domain Generalization by Seeking Flat Minima

Paper: "Domain Generalization by Seeking Flat Minima" (NeurIPS 2021)
Official source: https://github.com/khanrc/swad

Key Idea: Average model weights during training using a "loss valley" 
detection strategy to find flat minima that generalize better.
Adapted from: khanrc/swad/domainbed/swad.py + domainbed/lib/swa_utils.py
"""

from typing import Dict, List, Optional
from copy import deepcopy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DGMethod


class AveragedModel(nn.Module):
    """
    Running average of model parameters.
    Adapted from khanrc/swad/domainbed/lib/swa_utils.py::AveragedModel.
    """

    def __init__(self, backbone, classifier, device=None):
        super().__init__()
        self.backbone = deepcopy(backbone)
        self.classifier = deepcopy(classifier)
        if device is not None:
            self.backbone = self.backbone.to(device)
            self.classifier = self.classifier.to(device)
        self.n_averaged = 0
        self.start_step = -1
        self.end_step = -1
        self.end_loss = None

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def update_parameters(self, backbone, classifier, step=None,
                          start_step=None, end_step=None):
        """Update averaged parameters using running mean."""
        for p_avg, p in zip(self.backbone.parameters(), backbone.parameters()):
            device = p_avg.device
            p_model = p.detach().to(device)
            if self.n_averaged == 0:
                p_avg.detach().copy_(p_model)
            else:
                p_avg.detach().copy_(
                    p_avg.detach() + (p_model - p_avg.detach()) / (self.n_averaged + 1)
                )

        for p_avg, p in zip(self.classifier.parameters(), classifier.parameters()):
            device = p_avg.device
            p_model = p.detach().to(device)
            if self.n_averaged == 0:
                p_avg.detach().copy_(p_model)
            else:
                p_avg.detach().copy_(
                    p_avg.detach() + (p_model - p_avg.detach()) / (self.n_averaged + 1)
                )

        self.n_averaged += 1

        if start_step is not None and self.n_averaged == 1:
            self.start_step = start_step
        if end_step is not None:
            self.end_step = end_step
        if step is not None:
            if self.n_averaged == 1:
                self.start_step = step
            self.end_step = step


class SWAD(DGMethod):
    """
    Stochastic Weight Averaging Densely (SWAD).

    Uses loss valley detection to decide when to start/stop weight 
    averaging, producing a flat-minimum model for better generalization.

    Adapted from khanrc/swad/domainbed/swad.py::LossValley.
    """

    NAME = "swad"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        n_converge: int = 3,        # Converge detector window size
        n_tolerance: int = 6,       # Loss min smoothing window size
        tolerance_ratio: float = 0.3,  # Dead valley threshold ratio
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        self.feature_dim = feature_dim

        # --- LossValley state (from official swad.py) ---
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio

        self.converge_Q = deque(maxlen=n_converge)
        self.smooth_Q = deque(maxlen=n_tolerance)

        self.final_model = None         # The averaged model output
        self.converge_step = None
        self.dead_valley = False
        self.threshold = None

        self.iteration = 0

    def _snapshot(self, val_loss):
        """Create a snapshot of current model as an AveragedModel."""
        snap = AveragedModel(self.backbone, self.classifier,
                             device=torch.device('cpu'))
        snap.start_step = self.iteration
        snap.end_step = self.iteration
        snap.end_loss = val_loss
        return snap

    def _update_loss_valley(self, val_loss: float):
        """
        Core LossValley logic from official swad.py::LossValley.update_and_evaluate.
        Called after each training step with the current training loss
        (used as a proxy for validation loss within observe).
        """
        if self.dead_valley:
            return

        frozen = self._snapshot(val_loss)
        self.converge_Q.append(frozen)
        self.smooth_Q.append(frozen)

        if self.converge_step is None:
            # Not yet converged
            if len(self.converge_Q) < self.n_converge:
                return

            losses = [m.end_loss for m in self.converge_Q]
            min_idx = int(np.argmin(losses))

            if min_idx == 0:
                # Converged: the oldest entry in window has min loss
                self.converge_step = self.converge_Q[0].end_step
                untilmin = self.converge_Q[min_idx]
                self.final_model = AveragedModel(
                    untilmin.backbone, untilmin.classifier,
                    device=torch.device('cpu')
                )
                self.final_model.start_step = untilmin.start_step

                th_base = float(np.mean(losses))
                self.threshold = th_base * (1.0 + self.tolerance_ratio)

                # Absorb remaining converge_Q entries
                for i in range(1, len(self.converge_Q)):
                    model = self.converge_Q[i]
                    self.final_model.update_parameters(
                        model.backbone, model.classifier,
                        start_step=model.start_step,
                        end_step=model.end_step,
                    )
            return

        # Already converged => check loss valley
        if self.smooth_Q[0].end_step < self.converge_step:
            return

        min_vloss = min(m.end_loss for m in self.smooth_Q)
        if min_vloss > self.threshold:
            self.dead_valley = True
            return

        model = self.smooth_Q[0]
        self.final_model.update_parameters(
            model.backbone, model.classifier,
            start_step=model.start_step,
            end_step=model.end_step,
        )

    def observe(
        self,
        x_domains: List[torch.Tensor],
        y_domains: List[torch.Tensor],
    ) -> Dict[str, float]:
        """Process training batch with loss-valley-based weight averaging."""
        self.iteration += 1

        total_loss = 0.0
        n_samples = 0

        for x, y in zip(x_domains, y_domains):
            x = x.to(self.device)
            y = y.to(self.device)

            f = self.backbone(x)
            if f.dim() > 2:
                f = F.adaptive_avg_pool2d(f, 1).flatten(1)

            logits = self.classifier(f)
            loss = F.cross_entropy(logits, y)

            total_loss += loss * x.size(0)
            n_samples += x.size(0)

        avg_loss = total_loss / n_samples

        # Backward pass
        avg_loss.backward()

        # Update loss valley detector
        self._update_loss_valley(avg_loss.item())

        return {
            'total_loss': avg_loss.item(),
            'converged': self.converge_step is not None,
            'dead_valley': self.dead_valley,
            'n_averaged': self.final_model.n_averaged if self.final_model else 0,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use averaged model for inference if available."""
        x = x.to(self.device)

        if self.final_model is not None:
            self.final_model = self.final_model.to(self.device)
            out = self.final_model(x)
            self.final_model = self.final_model.cpu()
            return out

        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        if self.final_model is not None:
            self.final_model = self.final_model.to(self.device)
            f = self.final_model.backbone(x)
            self.final_model = self.final_model.cpu()
        else:
            f = self.backbone(x)

        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f
