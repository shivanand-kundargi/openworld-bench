"""
Base class for Domain Generalization methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class DGMethod(ABC):
    """
    Abstract base class for Domain Generalization methods.
    
    DG methods train on multiple source domains and aim to generalize
    to an unseen target domain. Key difference from DA: no target data
    during training.
    
    All DG methods must implement:
    - observe(): Process batches from multiple source domains
    - get_features(): Extract features for analysis
    """
    
    NAME: str = "base_dg"
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        device: torch.device,
        **kwargs
    ):
        """
        Initialize DG method.
        
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
        x_domains: List[torch.Tensor],
        y_domains: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Process a training batch from multiple domains.
        
        Args:
            x_domains: List of images from each source domain
            y_domains: List of labels from each source domain
            
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
        }
        if self.classifier is not None:
            state['classifier'] = self.classifier.state_dict()
        return state
    
    def load_state_dict(self, state: Dict):
        """Load state dict from checkpoint."""
        self.backbone.load_state_dict(state['backbone'])
        if self.classifier is not None and 'classifier' in state:
            self.classifier.load_state_dict(state['classifier'])
