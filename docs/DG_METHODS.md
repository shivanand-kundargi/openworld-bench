# Domain Generalization Methods

## Classic Methods

### IRM (Invariant Risk Minimization)
**Paper**: Invariant Risk Minimization (ArXiv 2019)

**Key Idea**: Learn representations such that the optimal classifier on top is invariant across all training domains.

**Components**:
- Invariance penalty measuring classifier optimality across domains
- Penalty annealing for stable training

**Loss**: `L = Σ_d L_erm(d) + λ * Σ_d ||∇_w L_erm(d)||²`

---

### VREx (Variance Risk Extrapolation)
**Paper**: Out-of-Distribution Generalization via Risk Extrapolation (ICML 2021)

**Key Idea**: Minimize the variance of per-domain risks to encourage learning features with consistent predictive quality.

**Components**:
- Per-domain loss computation
- Variance penalty across domains

**Loss**: `L = mean(L_domains) + λ * var(L_domains)`

---

### CORAL (Correlation Alignment)
**Paper**: Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016)

**Key Idea**: Align second-order statistics (covariance matrices) of feature distributions across domains.

**Components**:
- Covariance computation per domain
- Frobenius norm distance between covariances

**Loss**: `L = L_task + λ * ||C_source - C_target||²_F`

---

## Recent Methods (To Be Added)

### SWAD (NeurIPS 2021)
*Implementation pending*

### MIRO (ECCV 2022)
*Implementation pending*

### EoA (ICLR 2023)
*Implementation pending*

---

## Implementation Files

- `openworld/methods/dg/base.py` - Base class
- `openworld/methods/dg/irm.py` - IRM implementation
- `openworld/methods/dg/vrex.py` - VREx implementation
- `openworld/methods/dg/coral.py` - CORAL implementation
