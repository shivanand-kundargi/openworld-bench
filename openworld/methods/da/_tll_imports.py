"""
Helper module for importing Transfer-Learning-Library (TLL) components.

TLL provides official, well-tested implementations of:
- Domain Adversarial Loss (DANN)
- Conditional Domain Adversarial Loss (CDAN)
- Maximum Classifier Discrepancy (MCD)
- CORAL Loss
- Gradient Reversal Layer (warm-start)
- Domain Discriminator

Reference: https://github.com/thuml/Transfer-Learning-Library
"""

import sys
import os

# Add TLL to path
_TLL_ROOT = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', 'Transfer-Learning-Library'
)
_TLL_ROOT = os.path.abspath(_TLL_ROOT)

if _TLL_ROOT not in sys.path:
    sys.path.insert(0, _TLL_ROOT)

# GRL
from tllib.modules.grl import WarmStartGradientReverseLayer, GradientReverseFunction

# Domain Discriminator
from tllib.modules.domain_discriminator import DomainDiscriminator

# DANN
from tllib.alignment.dann import DomainAdversarialLoss

# CDAN
from tllib.alignment.cdan import (
    ConditionalDomainAdversarialLoss,
    RandomizedMultiLinearMap,
    MultiLinearMap,
)

# MCD
from tllib.alignment.mcd import classifier_discrepancy, ImageClassifierHead

# CORAL
from tllib.alignment.coral import CorrelationAlignmentLoss
