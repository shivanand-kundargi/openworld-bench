"""
Continual Learning Metrics

Metrics:
- Average Accuracy: Mean accuracy across all tasks after final task
- Forgetting: Average drop in accuracy on old tasks
- Forward Transfer (FWT): Performance on new tasks before learning them
- Backward Transfer (BWT): Performance change on old tasks after learning new ones
"""

from typing import Dict, List, Optional
import numpy as np


class CLMetrics:
    """Compute standard Continual Learning metrics."""
    
    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self.accuracy_matrix = np.zeros((n_tasks, n_tasks))
        
    def update(self, trained_task: int, eval_task: int, accuracy: float):
        """Update accuracy matrix."""
        self.accuracy_matrix[trained_task, eval_task] = accuracy
    
    def average_accuracy(self) -> float:
        """Average accuracy after training on all tasks."""
        return np.mean(self.accuracy_matrix[-1, :])
    
    def forgetting(self) -> float:
        """Average forgetting across tasks."""
        if self.n_tasks <= 1:
            return 0.0
        forgetting_values = []
        for j in range(self.n_tasks - 1):
            max_acc = np.max(self.accuracy_matrix[j:self.n_tasks-1, j])
            final_acc = self.accuracy_matrix[-1, j]
            forgetting_values.append(max_acc - final_acc)
        return np.mean(forgetting_values)
    
    def backward_transfer(self) -> float:
        """Backward Transfer (BWT)."""
        if self.n_tasks <= 1:
            return 0.0
        bwt_values = []
        for j in range(self.n_tasks - 1):
            bwt_values.append(self.accuracy_matrix[-1, j] - self.accuracy_matrix[j, j])
        return np.mean(bwt_values)
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all CL metrics."""
        return {
            'avg_accuracy': self.average_accuracy(),
            'forgetting': self.forgetting(),
            'bwt': self.backward_transfer(),
        }
    
    def get_accuracy_matrix(self) -> np.ndarray:
        """Return the accuracy matrix."""
        return self.accuracy_matrix.copy()
