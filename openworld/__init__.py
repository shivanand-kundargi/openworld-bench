"""
openworld-bench: Cross-Setting Evaluation Benchmark for DA, DG, and CL Methods

This package provides a unified framework to evaluate how methods designed for
one learning paradigm (Domain Adaptation, Domain Generalization, or Continual Learning)
perform under different settings.
"""

__version__ = "0.1.0"

from . import methods
from . import datasets
from . import protocols
from . import metrics
from . import utils
