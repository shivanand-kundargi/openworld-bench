"""
CUB-200-2011 Dataset

Caltech-UCSD Birds-200-2011: Fine-grained bird classification.
200 bird species, ~12,000 images.

For CL: Split into sequential tasks by class.

Download: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
"""

import os
from typing import List, Optional, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np


class CUB200(Dataset):
    """CUB-200-2011 dataset."""
    
    NUM_CLASSES = 200
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root: Path to CUB_200_2011 directory
            train: Whether to load training set
            transform: Image transforms
        """
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
        """Load image paths and labels."""
        images_txt = os.path.join(self.root, 'images.txt')
        labels_txt = os.path.join(self.root, 'image_class_labels.txt')
        split_txt = os.path.join(self.root, 'train_test_split.txt')
        
        if all(os.path.exists(f) for f in [images_txt, labels_txt, split_txt]):
            # Load from official txt files
            images = {}
            with open(images_txt, 'r') as f:
                for line in f:
                    idx, path = line.strip().split()
                    images[int(idx)] = path
            
            labels = {}
            with open(labels_txt, 'r') as f:
                for line in f:
                    idx, label = line.strip().split()
                    labels[int(idx)] = int(label) - 1  # 0-indexed
            
            splits = {}
            with open(split_txt, 'r') as f:
                for line in f:
                    idx, is_train = line.strip().split()
                    splits[int(idx)] = int(is_train)
            
            for idx in images:
                is_train = splits.get(idx, 1)
                if (self.train and is_train) or (not self.train and not is_train):
                    img_path = os.path.join(self.root, 'images', images[idx])
                    label = labels[idx]
                    self.samples.append((img_path, label))
                    self.targets.append(label)
        else:
            # Fallback: scan directory structure
            images_dir = os.path.join(self.root, 'images')
            if os.path.exists(images_dir):
                class_dirs = sorted(os.listdir(images_dir))
                for class_idx, class_dir in enumerate(class_dirs):
                    class_path = os.path.join(images_dir, class_dir)
                    if os.path.isdir(class_path):
                        images = sorted([f for f in os.listdir(class_path) 
                                        if f.endswith(('.jpg', '.png'))])
                        n_train = int(len(images) * 0.8)
                        
                        if self.train:
                            images = images[:n_train]
                        else:
                            images = images[n_train:]
                        
                        for img_name in images:
                            img_path = os.path.join(class_path, img_name)
                            self.samples.append((img_path, class_idx))
                            self.targets.append(class_idx)
        
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


class SequentialCUB200:
    """Sequential task-based wrapper for CUB-200."""
    
    def __init__(
        self,
        root: str,
        n_tasks: int = 10,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.train_dataset = CUB200(root, train=True)
        self.test_dataset = CUB200(root, train=False)
        
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.n_classes = 200
        self.classes_per_task = self.n_classes // n_tasks
        
        self.train_task_indices = self._create_task_splits(self.train_dataset)
        self.test_task_indices = self._create_task_splits(self.test_dataset)
        
    def _create_task_splits(self, dataset) -> List[List[int]]:
        """Split dataset indices by task."""
        task_indices = []
        targets = dataset.targets
        
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
        if train:
            indices = self.train_task_indices[task_id]
            dataset = self.train_dataset
        else:
            indices = self.test_task_indices[task_id]
            dataset = self.test_dataset
        
        subset = Subset(dataset, indices)
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
