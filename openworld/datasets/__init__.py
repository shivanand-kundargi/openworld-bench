"""
Dataset loaders for openworld-bench.

DA/DG Datasets:
- DomainNet: Large-scale multi-domain dataset (6 domains, 345 classes)
- Office-Home: Standard DA benchmark (4 domains, 65 classes)

CL Datasets:
- ImageNet-R: ImageNet Renditions (200 classes)
- CUB-200: Fine-grained bird classification (200 classes)
- Stanford Cars: Fine-grained car classification (196 classes)
- iNaturalist: Fine-grained species classification (configurable)
"""

from .domainnet import DomainNet, get_domainnet_loaders
from .office_home import OfficeHome, get_office_home_loaders
from .imagenet_r import ImageNetR, SequentialImageNetR
from .cub200 import CUB200, SequentialCUB200
from .stanford_cars import StanfordCars, SequentialStanfordCars
from .inaturalist import INaturalist, SequentialINaturalist

__all__ = [
    # DA/DG datasets
    'DomainNet',
    'get_domainnet_loaders',
    'OfficeHome',
    'get_office_home_loaders',
    
    # CL datasets (base)
    'ImageNetR',
    'CUB200',
    'StanfordCars',
    'INaturalist',
    
    # CL datasets (sequential wrappers)
    'SequentialImageNetR',
    'SequentialCUB200',
    'SequentialStanfordCars',
    'SequentialINaturalist',
]

# Dataset registry
DATASETS = {
    'domainnet': DomainNet,
    'office_home': OfficeHome,
    'imagenet_r': ImageNetR,
    'cub200': CUB200,
    'stanford_cars': StanfordCars,
    'inaturalist': INaturalist,
}

SEQUENTIAL_DATASETS = {
    'imagenet_r': SequentialImageNetR,
    'cub200': SequentialCUB200,
    'stanford_cars': SequentialStanfordCars,
    'inaturalist': SequentialINaturalist,
}


def get_dataset(name: str, root: str, **kwargs):
    """Get dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name](root, **kwargs)


def get_sequential_dataset(name: str, root: str, **kwargs):
    """Get sequential CL dataset by name."""
    if name not in SEQUENTIAL_DATASETS:
        raise ValueError(f"Unknown sequential dataset: {name}. Available: {list(SEQUENTIAL_DATASETS.keys())}")
    return SEQUENTIAL_DATASETS[name](root, **kwargs)
