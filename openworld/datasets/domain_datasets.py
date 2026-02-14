"""
Domain datasets for Domain Adaptation and Domain Generalization.

Supports:
- Office-Home (4 domains: Art, Clipart, Product, Real_World, 65 classes)
- DomainNet (6 domains: clipart, infograph, painting, quickdraw, real, sketch, 345 classes)

Auto-discovers datasets from known paths on disk.
"""

import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image

# ========== Known data paths ==========
# These are searched in order until a valid path is found.

OFFICE_HOME_CANDIDATES = [
    '/umbc/rs/pi_gokhale/users/shivank2/shivanand/X-Adapt/X-Adapt/data/OfficeHomeDataset_10072016',
]

DOMAINNET_CANDIDATES = [
    '/umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/data/DomainNet',
    '/umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/data/DomainNet',
    '/umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth_dev/data/DomainNet',
]

def _find_path(candidates, required_subdir=None):
    """Find first existing path from candidates."""
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            if required_subdir:
                if os.path.isdir(os.path.join(path, required_subdir)):
                    return path
            else:
                # Check it has at least some subdirectories
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if len(subdirs) > 0:
                    return path
    return None


class DomainImageFolder:
    """
    Wrapper around torchvision.ImageFolder for a single domain.
    Expects: root/class_name/image.jpg
    """
    def __init__(self, domain_path: str, transform=None):
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"Domain path not found: {domain_path}")
        self.dataset = datasets.ImageFolder(domain_path, transform=transform)
        self.domain_path = domain_path
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    @property
    def targets(self):
        return self.dataset.targets
    
    @property
    def classes(self):
        return self.dataset.classes


class MultiDomainDataset:
    """
    Generic multi-domain dataset for DA/DG settings.
    
    Loads each domain as a separate ImageFolder and provides
    combined or per-domain data loaders.
    """
    
    def __init__(
        self,
        root: str,
        source_domains: List[str],
        target_domain: str,
        train_transform=None,
        test_transform=None,
    ):
        self.root = root
        self.source_domains = source_domains
        self.target_domain = target_domain
        
        # Load source domain datasets
        self.source_datasets = {}
        for domain in source_domains:
            domain_path = os.path.join(root, domain)
            if not os.path.exists(domain_path):
                raise FileNotFoundError(
                    f"Source domain '{domain}' not found at {domain_path}"
                )
            self.source_datasets[domain] = DomainImageFolder(
                domain_path, transform=train_transform
            )
            print(f"  [Source] {domain}: {len(self.source_datasets[domain])} samples")
        
        # Load target domain dataset
        target_path = os.path.join(root, target_domain)
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Target domain '{target_domain}' not found at {target_path}"
            )
        self.target_dataset = DomainImageFolder(
            target_path, transform=test_transform or train_transform
        )
        print(f"  [Target] {target_domain}: {len(self.target_dataset)} samples")
        
        # Count classes from first source domain
        self.num_classes = len(self.source_datasets[source_domains[0]].classes)
    
    def get_source_loaders(self, batch_size: int, num_workers: int = 4) -> Dict[str, DataLoader]:
        """Get per-domain source data loaders."""
        loaders = {}
        for domain, dataset in self.source_datasets.items():
            loaders[domain] = DataLoader(
                dataset.dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True
            )
        return loaders
    
    def get_combined_source_loader(self, batch_size: int, num_workers: int = 4) -> DataLoader:
        """Get combined source data loader (all source domains)."""
        combined = ConcatDataset([d.dataset for d in self.source_datasets.values()])
        return DataLoader(
            combined, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
    
    def get_target_loader(self, batch_size: int, num_workers: int = 4) -> DataLoader:
        """Get target domain data loader."""
        return DataLoader(
            self.target_dataset.dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )


# ========== Dataset-specific classes ==========

class OfficeHomeDataset(MultiDomainDataset):
    """
    Office-Home: 4 domains, 65 classes.
    Domains: Art, Clipart, Product, Real_World
    """
    DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']
    NUM_CLASSES = 65
    
    def __init__(self, source_domains, target_domain, train_transform=None, test_transform=None):
        root = _find_path(OFFICE_HOME_CANDIDATES, required_subdir='Art')
        if root is None:
            raise FileNotFoundError(
                f"Office-Home dataset not found. Searched: {OFFICE_HOME_CANDIDATES}"
            )
        
        # Validate domains
        for d in source_domains + [target_domain]:
            if d not in self.DOMAINS:
                raise ValueError(f"Unknown Office-Home domain: {d}. Valid: {self.DOMAINS}")
        
        print(f"[OfficeHome] Found at: {root}")
        super().__init__(root, source_domains, target_domain, train_transform, test_transform)


class DomainNetDataset(MultiDomainDataset):
    """
    DomainNet: 6 domains, 345 classes.
    Domains: clipart, infograph, painting, quickdraw, real, sketch
    """
    DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    NUM_CLASSES = 345
    
    def __init__(self, source_domains, target_domain, train_transform=None, test_transform=None):
        root = _find_path(DOMAINNET_CANDIDATES, required_subdir=source_domains[0])
        if root is None:
            raise FileNotFoundError(
                "DomainNet dataset not found. The domain image folders need to be extracted. "
                f"Searched: {DOMAINNET_CANDIDATES}"
            )
        
        # Validate domains
        for d in source_domains + [target_domain]:
            if d not in self.DOMAINS:
                raise ValueError(f"Unknown DomainNet domain: {d}. Valid: {self.DOMAINS}")
        
        print(f"[DomainNet] Found at: {root}")
        super().__init__(root, source_domains, target_domain, train_transform, test_transform)


# ========== Factory function ==========

def get_domain_dataset(
    dataset_name: str,
    source_domains: List[str],
    target_domain: str,
    train_transform=None,
    test_transform=None,
):
    """
    Get domain dataset by name.
    
    Args:
        dataset_name: 'domainnet' or 'office_home'
        source_domains: List of source domain names
        target_domain: Target domain name
        train_transform: Training image transforms
        test_transform: Test image transforms
    
    Returns:
        MultiDomainDataset with source/target loaders
    """
    if dataset_name == 'office_home':
        return OfficeHomeDataset(source_domains, target_domain, train_transform, test_transform)
    elif dataset_name == 'domainnet':
        return DomainNetDataset(source_domains, target_domain, train_transform, test_transform)
    else:
        raise ValueError(f"Unknown domain dataset: {dataset_name}. Use 'office_home' or 'domainnet'")


def get_da_transforms():
    """Standard transforms for DA/DG datasets."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform
