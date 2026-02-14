"""
Stanford Cars Dataset

Fine-grained car classification with 196 classes.

Download: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
"""

import os
from typing import List, Optional, Tuple
from PIL import Image
import scipy.io

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np


class StanfordCars(Dataset):
    """Stanford Cars dataset."""
    
    NUM_CLASSES = 196
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = root
        self.train = train
        self.transform = transform or self._default_transform()
        
        self.samples = []
        self.targets = []
        
        self._load_data()
        
    def _default_transform(self):
        if self.train:
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
        """Load from .mat annotation files or directory structure."""
        split = 'train' if self.train else 'test'
        
        # Try loading from .mat file
        mat_file = os.path.join(self.root, f'cars_{split}_annos.mat')
        if os.path.exists(mat_file):
            try:
                annos = scipy.io.loadmat(mat_file)['annotations'][0]
                images_dir = os.path.join(self.root, f'cars_{split}')
                
                for anno in annos:
                    img_name = anno['fname'][0]
                    label = anno['class'][0, 0] - 1  # 0-indexed
                    img_path = os.path.join(images_dir, img_name)
                    
                    if os.path.exists(img_path):
                        self.samples.append((img_path, label))
                        self.targets.append(label)
            except Exception:
                pass
        
        # Fallback: directory structure
        if not self.samples:
            images_dir = os.path.join(self.root, 'cars_train' if self.train else 'cars_test')
            if os.path.exists(images_dir):
                # Assume subdirectory structure by class
                if os.path.isdir(os.path.join(images_dir, os.listdir(images_dir)[0] if os.listdir(images_dir) else '')):
                    for class_idx, class_dir in enumerate(sorted(os.listdir(images_dir))):
                        class_path = os.path.join(images_dir, class_dir)
                        if os.path.isdir(class_path):
                            for img_name in os.listdir(class_path):
                                if img_name.endswith(('.jpg', '.png')):
                                    img_path = os.path.join(class_path, img_name)
                                    self.samples.append((img_path, class_idx))
                                    self.targets.append(class_idx)
        
        self.targets = np.array(self.targets) if self.targets else np.array([])
    
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


class SequentialStanfordCars:
    """Sequential task-based wrapper for Stanford Cars."""
    
    def __init__(
        self,
        root: str,
        n_tasks: int = 10,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.train_dataset = StanfordCars(root, train=True)
        self.test_dataset = StanfordCars(root, train=False)
        
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.n_classes = 196
        self.classes_per_task = self.n_classes // n_tasks
        
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
            indices = [0]  # Avoid empty dataset
        
        subset = Subset(dataset, indices)
        return DataLoader(
            subset, batch_size=self.batch_size,
            shuffle=train, num_workers=self.num_workers,
            pin_memory=True, drop_last=False
        )
