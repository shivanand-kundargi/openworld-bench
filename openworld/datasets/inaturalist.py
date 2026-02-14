"""
iNaturalist Dataset Subset for Continual Learning

Uses a subset of iNaturalist for fine-grained species classification.

Download: https://github.com/visipedia/inat_comp
"""

import os
import json
from typing import List, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np


class INaturalist(Dataset):
    """iNaturalist dataset subset."""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_classes: int = 500,  # Limit classes for CL experiments
    ):
        self.root = root
        self.split = split
        self.transform = transform or self._default_transform()
        self.max_classes = max_classes
        
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        
        self._load_data()
        
    def _default_transform(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def _load_data(self):
        """Load from annotations or directory structure."""
        # Try JSON annotations
        anno_file = os.path.join(self.root, f'{self.split}2019.json')
        if os.path.exists(anno_file):
            try:
                with open(anno_file, 'r') as f:
                    data = json.load(f)
                
                # Build category mapping
                categories = {c['id']: i for i, c in enumerate(data['categories'][:self.max_classes])}
                
                for anno in data['annotations']:
                    if anno['category_id'] in categories:
                        img_id = anno['image_id']
                        img_info = next(i for i in data['images'] if i['id'] == img_id)
                        img_path = os.path.join(self.root, img_info['file_name'])
                        label = categories[anno['category_id']]
                        
                        if os.path.exists(img_path):
                            self.samples.append((img_path, label))
                            self.targets.append(label)
            except Exception:
                pass
        
        # Fallback: directory structure
        if not self.samples:
            images_dir = os.path.join(self.root, self.split)
            if os.path.exists(images_dir):
                class_dirs = sorted(os.listdir(images_dir))[:self.max_classes]
                self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}
                
                for class_name in class_dirs:
                    class_path = os.path.join(images_dir, class_name)
                    if os.path.isdir(class_path):
                        label = self.class_to_idx[class_name]
                        for img_name in os.listdir(class_path):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(class_path, img_name)
                                self.samples.append((img_path, label))
                                self.targets.append(label)
        
        self.targets = np.array(self.targets) if self.targets else np.array([])
        self.n_classes = len(set(self.targets)) if len(self.targets) > 0 else self.max_classes
    
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


class SequentialINaturalist:
    """Sequential task-based wrapper for iNaturalist."""
    
    def __init__(
        self,
        root: str,
        n_tasks: int = 10,
        batch_size: int = 32,
        num_workers: int = 4,
        max_classes: int = 500,
    ):
        self.train_dataset = INaturalist(root, split='train', max_classes=max_classes)
        self.test_dataset = INaturalist(root, split='val', max_classes=max_classes)
        
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.n_classes = self.train_dataset.n_classes
        self.classes_per_task = max(1, self.n_classes // n_tasks)
        
        self.train_task_indices = self._create_task_splits(self.train_dataset)
        self.test_task_indices = self._create_task_splits(self.test_dataset)
        
    def _create_task_splits(self, dataset) -> List[List[int]]:
        task_indices = []
        targets = dataset.targets
        
        if len(targets) == 0:
            return [[] for _ in range(self.n_tasks)]
        
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
        if train:
            indices = self.train_task_indices[task_id]
            dataset = self.train_dataset
        else:
            indices = self.test_task_indices[task_id]
            dataset = self.test_dataset
        
        if not indices:
            indices = [0]
        
        subset = Subset(dataset, indices)
        return DataLoader(
            subset, batch_size=self.batch_size,
            shuffle=train, num_workers=self.num_workers,
            pin_memory=True, drop_last=False
        )
