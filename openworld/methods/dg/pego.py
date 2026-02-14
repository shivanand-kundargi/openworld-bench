"""
PEGO: Parameter-Efficient Group with Orthogonal Regularization for Domain Generalization

Paper: "Parameter-Efficient Group with Orthogonal Regularization for Domain Generalization" (ECCV 2024)
Authors: JudgingH et al.

Key Idea:
1.  **LoRA**: Use Low-Rank Adaptation for parameter efficiency.
2.  **Group Orthogonal Regularization**: Enforce orthogonality between LoRA modules of different domains/groups to promote diverse feature learning.
3.  **DomainBed**: Built on DomainBed framework.

Implementation:
We will simulate the PEGO logic:
- In `__init__`: Inject LoRA modules into backbone (if supported) or use a simplified adapter approach.
- In `observe`: Compute orthogonality loss between adapters.

Since injecting LoRA into an arbitrary backbone without a library like `peft` is complex,
we will implement a "PEGO-Lite" that adds parallel adapter layers to the *classifier* or *penultimate* features, 
and regularizes them. 
Authentic PEGO requires modifying the backbone layers (Attention/Linear).
We will assume the backbone is frozen and we learn adapters on top, or we wrap standard Linear layers?
Given the constraints, we will implement the **Orthogonal Regularization** logic on a set of learnable adapter banks.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import DGMethod


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class PEGO(DGMethod):
    """
    PEGO (ECCV 2024) - Simplified Port.
    """
    NAME = "pego"

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        num_groups: int = 4, # Number of orthogonal groups (conceptually domains)
        rank: int = 4,
        ortho_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)
        
        self.num_groups = num_groups
        self.ortho_weight = ortho_weight
        self.feature_dim = feature_dim
        
        # In full PEGO, LoRA is applied to every layer.
        # Here, we apply it to a projection head after the backbone to simulate "adaptation".
        # This is strictly less powerful but fits the interface without rewriting backbones.
        
        # Group of LoRA Adapters
        self.adapters = nn.ModuleList([
            LoRALayer(feature_dim, feature_dim, rank=rank)
            for _ in range(num_groups)
        ])
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.to(device)

    def _ortho_loss(self):
        # Orthogonality between LoRA matrices B of different groups
        # B matrix: [out, rank]
        # We want B_i and B_j to be orthogonal
        loss = 0.0
        cnt = 0
        for i in range(self.num_groups):
            for j in range(i + 1, self.num_groups):
                # Cosine similarity or inner product?
                # PEGO paper minimizes abs(cosine) or dot product
                
                # Flatten B: [out * rank]
                b_i = self.adapters[i].lora_B.flatten()
                b_j = self.adapters[j].lora_B.flatten()
                
                dot = torch.dot(b_i, b_j)
                norm_i = torch.norm(b_i) + 1e-6
                norm_j = torch.norm(b_j) + 1e-6
                
                loss += torch.abs(dot / (norm_i * norm_j))
                cnt += 1
                
        return loss / max(cnt, 1)

    def observe(
        self,
        x_domains: list,
        y_domains: list,
    ) -> Dict[str, float]:
        
        # Flatten domains for batch processing?
        # PEGO usually trains on all source domains.
        # We process each domain with *all* adapters (Ensemble) or specific?
        # PEGO is DG, so at test time we don't know domain.
        # Strategy: Average of all adapters output.
        
        total_cls_loss = 0.0
        
        for x, y in zip(x_domains, y_domains):
            x = x.to(self.device)
            y = y.to(self.device)
            
            with torch.no_grad():
                f = self.backbone(x)
                if f.dim() > 2: f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            
            # Forward through all adapters and average
            adapter_outs = [adapter(f) for adapter in self.adapters]
            f_adapted = sum(adapter_outs) / self.num_groups
            
            # Residual connection (LoRA adds to original)
            f_final = f + f_adapted
            
            logits = self.classifier(f_final)
            total_cls_loss += F.cross_entropy(logits, y)
            
        total_cls_loss /= len(x_domains)
        
        # Orthogonality
        ortho_l = self._ortho_loss()
        
        loss = total_cls_loss + self.ortho_weight * ortho_l
        loss.backward()
        
        return {
            'total_loss': loss.item(),
            'cls_loss': total_cls_loss.item(),
            'ortho_loss': ortho_l.item()
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2: f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        
        # Inference: Average all adapters
        adapter_outs = [adapter(f) for adapter in self.adapters]
        f_adapted = sum(adapter_outs) / self.num_groups
        
        return self.classifier(f + f_adapted)

    def parameters(self):
        # Optimize adapters + classifier
        return list(self.adapters.parameters()) + list(self.classifier.parameters())
