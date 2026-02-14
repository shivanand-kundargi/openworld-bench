"""
Domain Adaptation Metrics

Metrics:
- Target Accuracy: Classification accuracy on target domain
- Per-domain Accuracy: Accuracy breakdown by domain
"""

from typing import Dict, List
import numpy as np


class DAMetrics:
    """Compute standard Domain Adaptation metrics."""
    
    def __init__(self, domains: List[str]):
        self.domains = domains
        self.per_domain_accuracy = {d: 0.0 for d in domains}
        
    def update(self, domain: str, accuracy: float):
        """Update accuracy for a domain."""
        self.per_domain_accuracy[domain] = accuracy
    
    def target_accuracy(self, target_domain: str) -> float:
        """Get accuracy on target domain."""
        return self.per_domain_accuracy.get(target_domain, 0.0)
    
    def average_accuracy(self) -> float:
        """Average accuracy across all domains."""
        return np.mean(list(self.per_domain_accuracy.values()))
    
    def compute_all(self, target_domain: str) -> Dict[str, float]:
        """Compute all DA metrics."""
        return {
            'target_accuracy': self.target_accuracy(target_domain),
            'average_accuracy': self.average_accuracy(),
            'per_domain': dict(self.per_domain_accuracy),
        }
