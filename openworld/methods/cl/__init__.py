"""
Continual Learning Methods

Classic Methods:
- iCaRL: Incremental Classifier and Representation Learning (CVPR 2017)
- DER: Dark Experience Replay (NeurIPS 2020)
- LwF: Learning without Forgetting (ECCV 2016)

Recent Methods:
- CODA-Prompt: Continual Decomposed Attention-based Prompting (CVPR 2023)
- X-DER: Extended Dark Experience Replay (TPAMI 2023)
- MEMO: Memory Efficient Online Learning (ICLR 2024)
"""


from .base import CLMethod, Buffer
from .icarl import ICarl
from .coda_prompt import CodaPrompt
from .ewc import EWC
from .l2p import L2P
from .dualprompt import DualPrompt

__all__ = [
    'CLMethod',
    'Buffer',
    'ICarl',
    'CodaPrompt',
    'EWC',
    'L2P',
    'DualPrompt',
]

# Method registry
CL_METHODS = {
    'icarl': ICarl,
    'coda_prompt': CodaPrompt,
    'ewc': EWC,
    'l2p': L2P,
    'dualprompt': DualPrompt,
}
