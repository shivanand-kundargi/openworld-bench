"""
Base class for Continual Learning methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class Buffer:
    """
    Experience replay buffer for continual learning.
    
    Stores samples from previous tasks for replay during training
    on new tasks.
    """
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.examples = None
        self.labels = None
        self.logits = None
        self.task_labels = None
        
        self.num_seen_examples = 0
        
    def is_empty(self) -> bool:
        return self.num_seen_examples == 0

    @property
    def current_size(self) -> int:
        return min(self.num_seen_examples, self.buffer_size)
    
    def add_data(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None,
    ):
        """Add data to buffer using reservoir sampling."""
        batch_size = examples.size(0)
        
        for i in range(batch_size):
            idx = self.num_seen_examples
            
            if self.num_seen_examples < self.buffer_size:
                # Buffer not full, just add
                if self.examples is None:
                    self.examples = examples[i:i+1].detach().clone()
                    self.labels = labels[i:i+1].detach().clone()
                    if logits is not None:
                        self.logits = logits[i:i+1].detach().clone()
                    if task_labels is not None:
                        self.task_labels = task_labels[i:i+1].detach().clone()
                else:
                    self.examples = torch.cat([self.examples, examples[i:i+1].detach().clone()])
                    self.labels = torch.cat([self.labels, labels[i:i+1].detach().clone()])
                    if logits is not None and self.logits is not None:
                        self.logits = torch.cat([self.logits, logits[i:i+1].detach().clone()])
                    if task_labels is not None and self.task_labels is not None:
                        self.task_labels = torch.cat([self.task_labels, task_labels[i:i+1].detach().clone()])
            else:
                # Reservoir sampling
                rand_idx = torch.randint(0, self.num_seen_examples, (1,)).item()
                if rand_idx < self.buffer_size:
                    self.examples[rand_idx] = examples[i].detach().clone()
                    self.labels[rand_idx] = labels[i].detach().clone()
                    if logits is not None and self.logits is not None:
                        self.logits[rand_idx] = logits[i].detach().clone()
                    if task_labels is not None and self.task_labels is not None:
                        self.task_labels[rand_idx] = task_labels[i].detach().clone()
            
            self.num_seen_examples += 1
    
    def get_data(
        self,
        size: int,
        return_logits: bool = False,
        return_task_labels: bool = False,
    ) -> Tuple:
        """Get random batch from buffer."""
        if self.is_empty():
            return None
        
        current_size = min(self.num_seen_examples, self.buffer_size)
        indices = torch.randperm(current_size)[:min(size, current_size)]
        
        examples = self.examples[indices].to(self.device)
        labels = self.labels[indices].to(self.device)
        
        result = [examples, labels]
        
        if return_logits and self.logits is not None:
            result.append(self.logits[indices].to(self.device))
        if return_task_labels and self.task_labels is not None:
            result.append(self.task_labels[indices].to(self.device))
            
        return tuple(result)


class CLMethod(ABC):
    """
    Abstract base class for Continual Learning methods.
    
    CL methods learn from a sequence of tasks, with the goal of
    maintaining performance on previous tasks while learning new ones.
    
    All CL methods must implement:
    - observe(): Process a single training batch (potentially with replay)
    - begin_task(): Called at the start of each task
    - end_task(): Called at the end of each task
    """
    
    NAME: str = "base_cl"
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        buffer_size: int = 500,
        **kwargs
    ):
        """
        Initialize CL method.
        
        Args:
            backbone: Feature extractor network
            num_classes: Total number of classes across all tasks
            device: Torch device (cuda/cpu)
            buffer_size: Size of experience replay buffer
        """
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = device
        
        self.buffer = Buffer(buffer_size, device)
        self.classifier = None  # To be defined by subclasses
        
        self.current_task = 0
        self.n_seen_classes = 0
        
    @abstractmethod
    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
    ) -> Dict[str, float]:
        """
        Process a training batch.
        
        Args:
            x: Input images
            y: Labels
            task_id: Current task identifier
            
        Returns:
            Dictionary of losses for logging
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            x: Input images
            
        Returns:
            Class predictions
        """
        pass
    
    def begin_task(self, task_id: int, n_classes_in_task: int):
        """Called at the start of each task."""
        self.current_task = task_id
        self.n_seen_classes += n_classes_in_task
    
    def end_task(self, task_id: int):
        """Called at the end of each task."""
        pass
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        return self.backbone(x)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.backbone.train(mode)
        if self.classifier is not None:
            self.classifier.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def to(self, device: torch.device):
        """Move to device."""
        self.device = device
        self.backbone = self.backbone.to(device)
        if self.classifier is not None:
            self.classifier = self.classifier.to(device)
        return self
    
    def parameters(self):
        """Get all trainable parameters."""
        params = list(self.backbone.parameters())
        if self.classifier is not None:
            params += list(self.classifier.parameters())
        return params
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        state = {
            'backbone': self.backbone.state_dict(),
            'current_task': self.current_task,
            'n_seen_classes': self.n_seen_classes,
        }
        if self.classifier is not None:
            state['classifier'] = self.classifier.state_dict()
        return state
    
    def load_state_dict(self, state: Dict):
        """Load state dict from checkpoint."""
        self.backbone.load_state_dict(state['backbone'])
        self.current_task = state.get('current_task', 0)
        self.n_seen_classes = state.get('n_seen_classes', 0)
        if self.classifier is not None and 'classifier' in state:
            self.classifier.load_state_dict(state['classifier'])
