"""
EWC: Elastic Weight Consolidation

Paper: "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)
Authors: Kirkpatrick et al.

Implementation: Faithful port from mammoth (models/ewc_on.py).
Key logic:
1. After each task, compute Fisher Information Matrix (FIM) diagonal.
2. Store old parameters (star parameters).
3. Regularize training with penalty: sum(F_i * (theta - theta_star)^2)

Source: mammoth/models/ewc_on.py
"""

from typing import Dict, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CLMethod


class EWC(CLMethod):
    """
    Elastic Weight Consolidation (mammoth port).
    """

    NAME = "ewc"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_tasks: int,
        device: torch.device,
        feature_dim: int = 2048,
        e_lambda: float = 0.5,  # Regularization strength
        gamma: float = 1.0,     # Decay factor for previous FIMs (if online EWC)
        **kwargs
    ):
        super().__init__(backbone, num_classes, device, buffer_size=0)

        self.e_lambda = e_lambda
        self.gamma = gamma
        self.feature_dim = feature_dim

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)

        # EWC Storage
        self.log_soft = nn.LogSoftmax(dim=1)
        self.checkpoint = None  # Old model parameters
        self.fish = None        # Fisher Information Matrix

    def _get_logits(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward
        outputs = self._get_logits(x)
        loss = F.cross_entropy(outputs, y)

        # EWC Penalty
        penalty = torch.tensor(0.0, device=self.device)
        if self.fish is not None:
            for name, param in self.net.named_parameters():
                if name in self.fish:
                    # F * (theta - theta_star)^2
                    fisher = self.fish[name]
                    star_param = self.checkpoint[name]
                    penalty += (fisher * (param - star_param) ** 2).sum()

        total_loss = loss + self.e_lambda * penalty
        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'cls_loss': loss.item(),
            'ewc_loss': (self.e_lambda * penalty).item(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self._get_logits(x)

    def begin_task(self, task_id: int, num_classes_in_task: int = None):
        """Called at start of each task."""
        self.current_task = task_id
        # In standard EWC, we don't do much at start, 
        # but we need to ensure self.net refers to current model parts
        # For simplicity in loop, we create a property or just use methods
        pass

    @property
    def net(self):
        """Helper to get all trainable modules as one object-like structure for iteration"""
        # We need a way to iterate over backbone + classifier consistent with named_parameters
        # We can just return self which is a nn.Module containing them
        return self

    def end_task(self, task_id: int, train_loader):
        """
        Compute Fisher Information Matrix at end of task.
        """
        fish = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                fish[name] = torch.zeros_like(param, device=self.device)

        self.net.eval()
        
        # Compute Fisher
        # We need a small subset of data to compute Fisher
        # train_loader is passed for this purpose
        
        # In mammoth, they iterate through the loader
        for j, data in enumerate(train_loader):
            # Handle different data loader formats
            if len(data) == 2:
                inputs, labels = data
            elif len(data) == 3: # domain_idx included
                inputs, labels, _ = data
            else:
                inputs, labels = data[0], data[1]

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self._get_logits(inputs)
            # mammoth uses nll_loss on log_softmax
            loss = F.nll_loss(self.log_soft(outputs), labels)
            
            self.net.zero_grad()
            loss.backward()
            
            for name, param in self.net.named_parameters():
                if name in fish:
                    fish[name] += param.grad.data ** 2
            
            if j >= 50: # Limit samples for efficiency like mammoth often does
                break
        
        # Normalize
        for name in fish:
            fish[name] /= (j + 1)
        
        # Update Global Fisher (with gamma decay if online, or just sum)
        # Mammoth EWC (online) uses gamma to decay old fisher
        if self.fish is None:
            self.fish = fish
        else:
            for name in self.fish:
                self.fish[name] = self.gamma * self.fish[name] + fish[name]

        # Checkpoint current parameters
        self.checkpoint = {}
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.checkpoint[name] = param.data.clone()

        self.net.train()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.classifier.parameters())

    def state_dict(self) -> Dict:
        state = {
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'fish': self.fish,
            'checkpoint': self.checkpoint
        }
        return state

    def load_state_dict(self, state: Dict):
        self.backbone.load_state_dict(state['backbone'])
        self.classifier.load_state_dict(state['classifier'])
        self.fish = state.get('fish')
        self.checkpoint = state.get('checkpoint')
