"""
PMTrans: Prototypical Matching Transformer for Domain Adaptation

Paper: "Prototypical Matching Transformer" (CVPR 2023)

Key Idea: Use transformer attention to match target samples with
class prototypes computed from source domain.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import DAMethod


class PrototypeAttention(nn.Module):
    """Cross-attention between samples and class prototypes."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Args:
            query: [batch, dim] - samples
            key: [num_classes, dim] - prototypes  
            value: [num_classes, dim] - prototypes
        """
        B = query.size(0)
        N = key.size(0)
        
        q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(N, self.num_heads, self.head_dim)
        
        # Attention: [B, heads, N]
        attn = torch.einsum('bhd,nhd->bhn', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Output: [B, heads, dim]
        out = torch.einsum('bhn,nhd->bhd', attn, v)
        out = out.reshape(B, -1)
        
        return self.out_proj(out), attn


class PMTrans(DAMethod):
    """
    Prototypical Matching Transformer.
    
    Uses attention to match target samples with learned class prototypes.
    """
    
    NAME = "pmtrans"
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        bottleneck_dim: int = 256,
        trade_off: float = 1.0,
        num_heads: int = 4,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)
        
        self.trade_off = trade_off
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        ).to(device)
        
        # Class prototypes (learnable)
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, bottleneck_dim) * 0.01
        ).to(device)
        
        # Prototype attention
        self.proto_attn = PrototypeAttention(bottleneck_dim, num_heads).to(device)
        
        # Classifier
        self.classifier = nn.Linear(bottleneck_dim, num_classes).to(device)
        
        self.feature_dim = bottleneck_dim
        
    def _update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Update prototypes using EMA from source features."""
        with torch.no_grad():
            for c in labels.unique():
                mask = (labels == c)
                if mask.sum() > 0:
                    class_features = features[mask].mean(dim=0)
                    # EMA update
                    self.prototypes.data[c] = 0.9 * self.prototypes.data[c] + 0.1 * class_features
    
    def observe(
        self,
        x_source: torch.Tensor,
        y_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Dict[str, float]:
        """Process training batch with prototype matching."""
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
        
        # Update prototypes from source
        self._update_prototypes(b_source.detach(), y_source)
        
        # Source classification
        logits_source = self.classifier(b_source)
        cls_loss = F.cross_entropy(logits_source, y_source)
        
        # Prototype matching for target
        matched_target, attn_weights = self.proto_attn(
            b_target, self.prototypes, self.prototypes
        )
        
        # Prototype classification (pseudo-labels from attention)
        proto_logits = torch.cdist(b_target, self.prototypes, p=2)
        proto_probs = F.softmax(-proto_logits, dim=1)  # Negative distance
        
        # Entropy minimization on target
        entropy = -(proto_probs * torch.log(proto_probs + 1e-8)).sum(dim=1).mean()
        
        total_loss = cls_loss + self.trade_off * entropy
        
        return {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'entropy': entropy.item(),
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        b = self.bottleneck(f)
        return self.classifier(b)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.bottleneck(f)
    
    def parameters(self):
        return (
            list(self.backbone.parameters()) +
            list(self.bottleneck.parameters()) +
            list(self.classifier.parameters()) +
            list(self.proto_attn.parameters()) +
            [self.prototypes]
        )
