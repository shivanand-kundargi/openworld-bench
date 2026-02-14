"""
Domain Adaptation Methods

Classic Methods:
- DANN: Domain-Adversarial Training of Neural Networks (ICML 2015)
- CDAN: Conditional Adversarial Domain Adaptation (NeurIPS 2018)
- MCD: Maximum Classifier Discrepancy (CVPR 2018)

Recent Methods:
- ToAlign: Aligning Distributions Across Domains (ICLR 2023)
- PMTrans: Prototypical Matching Transformer (CVPR 2023)
"""

from .base import DAMethod
from .dann import DANN
from .cdan import CDAN
from .mcd import MCD
from .toalign import ToAlign
from .pmtrans import PMTrans
from .dapl import DAPL

__all__ = [
    'DAMethod',
    'DANN',
    'CDAN',
    'MCD',
    'ToAlign',
    'PMTrans',
    'DAPL',
]

# Method registry
DA_METHODS = {
    'dann': DANN,
    'cdan': CDAN,
    'mcd': MCD,
    'toalign': ToAlign,
    'pmtrans': PMTrans,
    'dapl': DAPL,
}
