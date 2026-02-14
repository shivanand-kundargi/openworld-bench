# Domain Adaptation Methods

## Classic Methods

### DANN (Domain-Adversarial Neural Networks)
**Paper**: Domain-Adversarial Training of Neural Networks (ICML 2015)

**Key Idea**: Uses a gradient reversal layer to train a domain discriminator adversarially, encouraging the feature extractor to produce domain-invariant representations.

**Components**:
- Feature extractor (backbone)
- Task classifier
- Domain discriminator with gradient reversal

**Loss**: `L = L_task + λ * L_domain`

---

### CDAN (Conditional Domain Adversarial Network)
**Paper**: Conditional Adversarial Domain Adaptation (NeurIPS 2018)

**Key Idea**: Conditions domain discrimination on classifier predictions to capture class-wise domain shift.

**Components**:
- Randomized multilinear map for efficient conditioning
- Conditional domain discriminator

**Loss**: `L = L_task + λ * L_conditional_domain`

---

### MCD (Maximum Classifier Discrepancy)
**Paper**: Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (CVPR 2018)

**Key Idea**: Uses two classifiers to detect target samples far from source support via discrepancy maximization.

**Training Steps**:
1. Train both classifiers on source
2. Maximize classifier discrepancy on target
3. Minimize discrepancy by updating generator

---

## Recent Methods (To Be Added)

### ToAlign (ICLR 2023)
*Implementation pending*

### PMTrans (CVPR 2023)
*Implementation pending*

---

## Implementation Files

- `openworld/methods/da/base.py` - Base class
- `openworld/methods/da/dann.py` - DANN implementation
- `openworld/methods/da/cdan.py` - CDAN implementation
- `openworld/methods/da/mcd.py` - MCD implementation
