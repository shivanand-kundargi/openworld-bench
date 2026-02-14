"""
ImageNet-R Dataset for Continual Learning

ImageNet-Renditions contains 30,000 images of various artistic renditions
of 200 ImageNet classes. 

For CL: Split into sequential tasks by class.

Download: https://github.com/hendrycks/imagenet-r
"""

import os
from typing import List, Optional, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np


class ImageNetR(Dataset):
    """ImageNet-R dataset for Continual Learning."""
    
    NUM_CLASSES = 200
    
    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root: Path to imagenet-r directory
            transform: Image transforms
        """
        self.root = root
        self.transform = transform or self._default_transform()
        
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        self._load_data()
        
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_data(self):
        """Load image paths and labels."""
        class_names = sorted([d for d in os.listdir(self.root) 
                             if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        
        for class_name in class_names:
            class_dir = os.path.join(self.root, class_name)
            label = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, label))
                    self.targets.append(label)
        
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SequentialImageNetR:
    """Sequential task-based wrapper for ImageNet-R."""
    
    def __init__(
        self,
        root: str,
        n_tasks: int = 10,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.dataset = ImageNetR(root)
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.n_classes = self.dataset.NUM_CLASSES
        self.classes_per_task = self.n_classes // n_tasks
        
        # Create task splits
        self.task_indices = self._create_task_splits()
        
    def _create_task_splits(self) -> List[List[int]]:
        """Split dataset indices by task (sequential classes)."""
        task_indices = []
        targets = self.dataset.targets
        
        for task in range(self.n_tasks):
            start_class = task * self.classes_per_task
            end_class = (task + 1) * self.classes_per_task
            if task == self.n_tasks - 1:
                end_class = self.n_classes
            
            task_classes = list(range(start_class, end_class))
            indices = np.where(np.isin(targets, task_classes))[0].tolist()
            task_indices.append(indices)
        
        return task_indices
    
    def get_task_loader(self, task_id: int, train: bool = True) -> DataLoader:
        """Get data loader for a specific task."""
        indices = self.task_indices[task_id]
        
        # Simple 80/20 split
        n_train = int(len(indices) * 0.8)
        if train:
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]
        
        subset = Subset(self.dataset, indices)
        return DataLoader(
            subset, batch_size=self.batch_size,
            shuffle=train, num_workers=self.num_workers,
            pin_memory=True, drop_last=train
        )
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """Get class indices for a task."""
        start = task_id * self.classes_per_task
        end = (task_id + 1) * self.classes_per_task
        if task_id == self.n_tasks - 1:
            end = self.n_classes
        return list(range(start, end))
