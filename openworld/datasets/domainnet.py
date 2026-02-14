"""
DomainNet Dataset

A large-scale benchmark for domain adaptation with 6 domains:
- clipart, infograph, painting, quickdraw, real, sketch

345 classes, ~600k images total.

Download: http://ai.bu.edu/M3SDA/
"""

import os
from typing import List, Optional, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DomainNet(Dataset):
    """DomainNet dataset for DA/DG."""
    
    DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    NUM_CLASSES = 345
    
    def __init__(
        self,
        root: str,
        domains: List[str],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root: Path to DomainNet directory
            domains: List of domains to load
            split: 'train' or 'test'
            transform: Image transforms
        """
        self.root = root
        self.domains = domains
        self.split = split
        self.transform = transform or self._default_transform()
        
        # Load samples
        self.samples = []  # (path, label, domain_idx)
        self.class_names = []
        
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
        """Load image paths and labels from split files."""
        for domain_idx, domain in enumerate(self.domains):
            split_file = os.path.join(self.root, f'{domain}_{self.split}.txt')
            
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            img_path = os.path.join(self.root, parts[0])
                            label = int(parts[1])
                            self.samples.append((img_path, label, domain_idx))
            else:
                # Fallback: scan directory structure
                domain_dir = os.path.join(self.root, domain)
                if os.path.exists(domain_dir):
                    for class_idx, class_name in enumerate(sorted(os.listdir(domain_dir))):
                        class_dir = os.path.join(domain_dir, class_name)
                        if os.path.isdir(class_dir):
                            for img_name in os.listdir(class_dir):
                                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                                    img_path = os.path.join(class_dir, img_name)
                                    self.samples.append((img_path, class_idx, domain_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
        img_path, label, domain_idx = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return random image on error
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_idx


def get_domainnet_loaders(
    root: str,
    source_domains: List[str],
    target_domain: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Get data loaders for domain adaptation."""
    
    # Source loader
    source_dataset = DomainNet(root, source_domains, split='train')
    source_loader = DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    # Target loader (unlabeled)
    target_dataset = DomainNet(root, [target_domain], split='train')
    target_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    # Test loader (target test set)
    test_dataset = DomainNet(root, [target_domain], split='test')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return {
        'source': source_loader,
        'target': target_loader,
        'test': test_loader,
    }
