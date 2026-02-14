"""
Utility functions for openworld-bench.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging
from datetime import datetime


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """
    Get backbone network.
    
    Args:
        name: Backbone name (resnet18, resnet50, vit_base, etc.)
        pretrained: Whether to use pretrained weights
        
    Returns:
        Backbone network (without final classification layer)
    """
    import torchvision.models as models
    import timm
    
    if name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Identity()
        return model, 512
    elif name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Identity()
        return model, 2048
    elif name.startswith('vit'):
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
        return model, model.num_features
    else:
        raise ValueError(f"Unknown backbone: {name}")


def setup_logging(output_dir: str, name: str = 'train') -> logging.Logger:
    """
    Setup logging to file and console.
    
    Args:
        output_dir: Directory for log files
        name: Logger name
        
    Returns:
        Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'{name}_{timestamp}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    output_dir: str,
    name: str = 'checkpoint',
    extra: dict = None,
):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if extra:
        state.update(extra)
    
    path = os.path.join(output_dir, f'{name}_epoch{epoch}.pth')
    torch.save(state, path)
    return path


def load_checkpoint(model, optimizer, path: str):
    """Load model checkpoint."""
    state = torch.load(path)
    model.load_state_dict(state['model'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    return state.get('epoch', 0)
