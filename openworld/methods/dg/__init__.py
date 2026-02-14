"""
Domain Generalization Methods

Classic Methods:
- IRM: Invariant Risk Minimization (ArXiv 2019)
- VREx: Risk Extrapolation (ICML 2021)
- CORAL: Correlation Alignment (ECCV 2016)

Recent Methods:
- SWAD: Domain Generalization by Seeking (NeurIPS 2021)
- MIRO: Mutual Information Regularization (ECCV 2022)
- EoA: Ensemble of Averages (ICLR 2023)
"""

from .base import DGMethod
from .irm import IRM
from .vrex import VREx
from .coral import CORAL
from .swad import SWAD
from .miro import MIRO
from .eoa import EoA
from .pego import PEGO

__all__ = [
    'DGMethod',
    'IRM',
    'VREx',
    'CORAL',
    'SWAD',
    'MIRO',
    'EoA',
    'PEGO',
]

# Method registry
DG_METHODS = {
    'irm': IRM,
    'vrex': VREx,
    'coral': CORAL,
    'swad': SWAD,
    'miro': MIRO,
    'eoa': EoA,
    'pego': PEGO,
}
