"""
Office-Home Dataset

Standard benchmark for domain adaptation with 4 domains:
- Art, Clipart, Product, Real_World

65 classes, ~15,500 images total.

Download: https://www.hemanthdv.org/officeHomeDataset.html
"""

import os
from typing import List, Optional, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OfficeHome(Dataset):
    """Office-Home dataset for DA/DG."""
    
    DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']
    NUM_CLASSES = 65
    
    def __init__(
        self,
        root: str,
        domains: List[str],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        train_ratio: float = 0.8,
    ):
        """
        Args:
            root: Path to Office-Home directory
            domains: List of domains to load
            split: 'train' or 'test'
            transform: Image transforms
            train_ratio: Train/test split ratio
        """
        self.root = root
        self.domains = domains
        self.split = split
        self.train_ratio = train_ratio
        self.transform = transform or self._default_transform()
        
        self.samples = []
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
        """Load image paths and labels."""
        # Build class mapping
        all_classes = set()
        for domain in self.DOMAINS:
            domain_dir = os.path.join(self.root, domain)
            if os.path.exists(domain_dir):
                all_classes.update(os.listdir(domain_dir))
        
        self.class_to_idx = {c: i for i, c in enumerate(sorted(all_classes))}
        
        # Load samples for specified domains
        for domain_idx, domain in enumerate(self.domains):
            domain_dir = os.path.join(self.root, domain)
            if not os.path.exists(domain_dir):
                continue
                
            for class_name in os.listdir(domain_dir):
                if class_name not in self.class_to_idx:
                    continue
                    
                class_dir = os.path.join(domain_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                images = [f for f in os.listdir(class_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                # Split by indices
                n_train = int(len(images) * self.train_ratio)
                if self.split == 'train':
                    images = images[:n_train]
                else:
                    images = images[n_train:]
                
                for img_name in images:
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label, domain_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
        img_path, label, domain_idx = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_idx


def get_office_home_loaders(
    root: str,
    source_domains: List[str],
    target_domain: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Get data loaders for domain adaptation."""
    
    source_dataset = OfficeHome(root, source_domains, split='train')
    source_loader = DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    target_dataset = OfficeHome(root, [target_domain], split='train')
    target_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    test_dataset = OfficeHome(root, [target_domain], split='test')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return {
        'source': source_loader,
        'target': target_loader,
        'test': test_loader,
    }
