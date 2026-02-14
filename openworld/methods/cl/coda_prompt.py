"""
CODA-Prompt: COntinual Decomposed Attention-based Prompting

Paper: "CODA-Prompt: COntinual Decomposed Attention-based Prompting for
       Rehearsal-Free Continual Learning" (CVPR 2023)
Authors: Smith et al.

Implementation: Direct import from mammoth (models/coda_prompt.py + coda_prompt_utils/).

IMPORTANT: CODA-Prompt requires a custom ViT backbone (vit_base_patch16_224).
Prompts are prepended as key/value pairs at each transformer layer — this is 
fundamentally incompatible with generic CNN backbones. The mammoth Model class
internally creates its own ViT, so the backbone parameter is overridden.

Source: mammoth/models/coda_prompt.py, mammoth/models/coda_prompt_utils/{model.py, vit.py}
"""

import sys
import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CLMethod

# Add mammoth to path for importing coda_prompt_utils
_MAMMOTH_ROOT = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', 'mammoth'
)
_MAMMOTH_ROOT = os.path.abspath(_MAMMOTH_ROOT)
if _MAMMOTH_ROOT not in sys.path:
    sys.path.insert(0, _MAMMOTH_ROOT)

from models.coda_prompt_utils.model import Model as CodaModel


class CodaPrompt(CLMethod):
    """
    CODA-Prompt (mammoth port).

    Uses mammoth's full CodaModel which includes:
    - Custom ViT-B/16 backbone (pretrained on ImageNet-21k, finetuned on ImageNet-1k)
    - Per-layer prompt pool with attention-based selection
    - Gram-Schmidt orthogonalization for prompt components
    - Orthogonal penalty loss

    Note: The backbone parameter is ignored; CodaModel creates its own ViT.
    """

    NAME = "coda_prompt"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_tasks: int,
        device: torch.device,
        pool_size: int = 100,
        prompt_len: int = 8,
        mu: float = 0.0,
        virtual_bs_iterations: int = 1,
        **kwargs
    ):
        # Initialize base with buffer_size=0 (rehearsal-free)
        super().__init__(backbone, num_classes, device, buffer_size=0)

        self.mu = mu
        self.virtual_bs_iterations = virtual_bs_iterations
        self.current_task = 0
        self.cpt = num_classes // num_tasks

        # mammoth Model: creates its own ViT + prompt pool + classifier
        self.net = CodaModel(
            num_classes=num_classes,
            pt=True,
            prompt_param=[num_tasks, [pool_size, prompt_len, 0]]
        ).to(device)

        self.net.task_id = 0

        # Override backbone reference (for compatibility with base class)
        self.backbone = self.net.feat
        self.classifier = self.net.last

        # Task offsets for masking
        self.n_past_classes = 0
        self.n_seen_classes = 0

        self.epoch_iteration = 0

    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """
        mammoth CODA-Prompt observe.

        loss = CE(logits[:, offset1:offset2], labels - offset1) + mu * prompt_loss
        With offset masking: logits[:, :offset1] = -inf
        Supports virtual batch size accumulation.
        """
        x = x.to(self.device)
        y = y.to(self.device).long()

        # Forward pass (mammoth: self.net(inputs, train=True))
        logits, loss_prompt = self.net(x, train=True)
        loss_prompt = loss_prompt.sum()

        # Offset masking (mammoth CODA-Prompt)
        offset_1 = self.n_past_classes
        offset_2 = self.n_seen_classes

        logits = logits[:, :offset_2]
        logits[:, :offset_1] = -float('inf')

        # Classification loss
        loss_ce = F.cross_entropy(logits, y)

        # Total loss
        loss = loss_ce + self.mu * loss_prompt

        # Virtual batch size (mammoth)
        (loss / float(self.virtual_bs_iterations)).backward()

        self.epoch_iteration += 1

        return {
            'total_loss': loss.item(),
            'cls_loss': loss_ce.item(),
            'prompt_loss': loss_prompt.item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference (no prompt loss)."""
        x = x.to(self.device)
        out = self.net(x)
        return out[:, :self.n_seen_classes]

    def begin_task(self, task_id: int, num_classes_in_task: int = None):
        """
        mammoth begin_task: update task_id, re-init prompts, reset scheduler.
        """
        self.current_task = task_id
        if num_classes_in_task is not None:
            self.cpt = num_classes_in_task
        self.n_past_classes = task_id * self.cpt
        self.n_seen_classes = (task_id + 1) * self.cpt

        if task_id != 0:
            self.net.task_id = task_id
            self.net.prompt.process_task_count()

        self.epoch_iteration = 0

    def end_task(self, task_id: int, train_loader=None):
        """Called at end of each task."""
        pass

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get ViT CLS token features."""
        x = x.to(self.device)
        return self.net(x, pen=True)

    def parameters(self):
        """Only optimize prompt parameters + classifier (backbone is frozen)."""
        return list(self.net.prompt.parameters()) + list(self.net.last.parameters())

    def state_dict(self) -> Dict:
        return {'net': self.net.state_dict()}

    def load_state_dict(self, state: Dict):
        self.net.load_state_dict(state['net'])
