"""
EoA: Ensemble of Averages for Domain Generalization

Paper: "Model Ratatouille: Recycling Diverse Models for Out-of-Distribution
Generalization" (ICLR 2023)

Key Idea: Average weights of models fine-tuned on different domains,
then ensemble multiple such averages.
"""

from typing import Dict, List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DGMethod


class EoA(DGMethod):
    """
    Ensemble of Averages (EoA).
    
    Maintains domain-specific models and ensembles their predictions.
    """
    
    NAME = "eoa"
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        feature_dim: int = 2048,
        num_domains: int = 4,
        ensemble_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(backbone, num_classes, device)
        
        self.num_domains = num_domains
        self.ensemble_weight = ensemble_weight
        
        # Shared classifier
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        
        # Domain-specific classifiers (for diversity)
        self.domain_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_classes).to(device)
            for _ in range(num_domains)
        ])
        
        # Track which domain we're training on
        self.current_domain_idx = 0
        
        self.feature_dim = feature_dim
        
    def observe(
        self,
        x_domains: List[torch.Tensor],
        y_domains: List[torch.Tensor],
    ) -> Dict[str, float]:
        """Process training batch from multiple domains."""
        total_loss = 0.0
        n_samples = 0
        
        for domain_idx, (x, y) in enumerate(zip(x_domains, y_domains)):
            x = x.to(self.device)
            y = y.to(self.device)
            domain_idx = min(domain_idx, len(self.domain_classifiers) - 1)
            
            f = self.backbone(x)
            if f.dim() > 2:
                f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            
            # Shared classifier loss
            logits_shared = self.classifier(f)
            loss_shared = F.cross_entropy(logits_shared, y)
            
            # Domain-specific classifier loss
            logits_domain = self.domain_classifiers[domain_idx](f)
            loss_domain = F.cross_entropy(logits_domain, y)
            
            # Combined loss
            loss = (1 - self.ensemble_weight) * loss_shared + self.ensemble_weight * loss_domain
            
            total_loss += loss * x.size(0)
            n_samples += x.size(0)
        
        avg_loss = total_loss / n_samples
        
        return {
            'total_loss': avg_loss.item(),
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction from all classifiers."""
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        
        # Average predictions from all classifiers
        logits_shared = self.classifier(f)
        logits_ensemble = logits_shared.clone()
        
        for domain_clf in self.domain_classifiers:
            logits_ensemble = logits_ensemble + domain_clf(f)
        
        # Average
        logits_ensemble = logits_ensemble / (1 + len(self.domain_classifiers))
        
        return logits_ensemble
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        f = self.backbone(x)
        if f.dim() > 2:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f
    
    def parameters(self):
        params = list(self.backbone.parameters()) + list(self.classifier.parameters())
        for clf in self.domain_classifiers:
            params += list(clf.parameters())
        return params
