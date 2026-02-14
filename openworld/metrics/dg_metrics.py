"""
Domain Generalization Metrics

Metrics:
- Leave-one-out Accuracy: Average accuracy when leaving each domain out
- Worst-domain Accuracy: Accuracy on the hardest domain
"""

from typing import Dict, List
import numpy as np


class DGMetrics:
    """Compute standard Domain Generalization metrics."""
    
    def __init__(self, domains: List[str]):
        self.domains = domains
        self.leave_one_out_accuracy = {d: 0.0 for d in domains}
        
    def update(self, held_out_domain: str, accuracy: float):
        """Update accuracy for a held-out domain."""
        self.leave_one_out_accuracy[held_out_domain] = accuracy
    
    def average_accuracy(self) -> float:
        """Average leave-one-out accuracy."""
        return np.mean(list(self.leave_one_out_accuracy.values()))
    
    def worst_domain_accuracy(self) -> float:
        """Accuracy on the worst domain."""
        return min(self.leave_one_out_accuracy.values())
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all DG metrics."""
        return {
            'average_accuracy': self.average_accuracy(),
            'worst_domain_accuracy': self.worst_domain_accuracy(),
            'per_domain': dict(self.leave_one_out_accuracy),
        }
