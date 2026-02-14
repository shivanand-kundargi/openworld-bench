"""
Base class for Domain Adaptation methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class DAMethod(ABC):
    """
    Abstract base class for Domain Adaptation methods.
    
    All DA methods must implement:
    - observe(): Process a batch from source and target domains
    - get_features(): Extract features for analysis
    """
    
    NAME: str = "base_da"
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        **kwargs
    ):
        """
        Initialize DA method.
        
        Args:
            backbone: Feature extractor network
            num_classes: Number of output classes
            device: Torch device (cuda/cpu)
        """
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = device
        self.classifier = None  # To be defined by subclasses
        
    @abstractmethod
    def observe(
        self,
        x_source: torch.Tensor,
        y_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Process a training batch.
        
        Args:
            x_source: Source domain images
            y_source: Source domain labels
            x_target: Target domain images (unlabeled)
            
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
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input.
        
        Args:
            x: Input images
            
        Returns:
            Feature representations
        """
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
        }
        if self.classifier is not None:
            state['classifier'] = self.classifier.state_dict()
        return state
    
    def load_state_dict(self, state: Dict):
        """Load state dict from checkpoint."""
        self.backbone.load_state_dict(state['backbone'])
        if self.classifier is not None and 'classifier' in state:
            self.classifier.load_state_dict(state['classifier'])
