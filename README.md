# openworld-bench

**A Unified Benchmark for Cross-Setting Evaluation of Domain Adaptation, Domain Generalization, and Continual Learning Methods**

## Overview

openworld-bench evaluates how methods designed for one learning paradigm (DA, DG, or CL) perform under different settings. This benchmark demonstrates that methods treating distributional shift and temporal non-stationarity as separate problems are insufficient for real-world open-world learning.

## Key Insight

| Method Origin | DA Setting | DG Setting | CL Setting |
|--------------|------------|------------|------------|
| **DA Methods** | ✅ Native | ❌ Cross | ❌ Cross |
| **DG Methods** | ❌ Cross | ✅ Native | ❌ Cross |
| **CL Methods** | ❌ Cross | ❌ Cross | ✅ Native |

**Hypothesis**: Methods designed for one setting will significantly underperform when evaluated in other settings, revealing the need for unified approaches.

---

## Supported Methods

### Domain Adaptation (DA)
| Method | Type | Paper |
|--------|------|-------|
| DANN | Classic | ICML 2015 |
| CDAN | Classic | NeurIPS 2018 |
| MCD | Classic | CVPR 2018 |
| ToAlign | Recent | ICLR 2023 |
| PMTrans | Recent | CVPR 2023 |

### Domain Generalization (DG)
| Method | Type | Paper |
|--------|------|-------|
| IRM | Classic | ArXiv 2019 |
| VREx | Classic | ICML 2021 |
| CORAL | Classic | ECCV 2016 |
| SWAD | Recent | NeurIPS 2021 |
| MIRO | Recent | ECCV 2022 |

### Continual Learning (CL)
| Method | Type | Paper |
|--------|------|-------|
| iCaRL | Classic | CVPR 2017 |
| DER | Classic | NeurIPS 2020 |
| LwF | Classic | ECCV 2016 |
| CODA-Prompt | Recent | CVPR 2023 |
| X-DER | Recent | TPAMI 2023 |

---

## Datasets

| Dataset | Classes | Domains/Tasks | Use Case |
|---------|---------|---------------|----------|
| DomainNet | 345 | 6 domains | DA/DG |
| Office-Home | 65 | 4 domains | DA/DG |
| ImageNet-R | 200 | 10 tasks | CL |
| CUB-200 | 200 | 10 tasks | CL/FGVC |
| Stanford Cars | 196 | - | FGVC |
| FGVC-Aircraft | 100 | - | FGVC |
| iNaturalist | 1000 | 10 tasks | CL/FGVC |

---

## Quick Start

### Installation

```bash
cd openworld-bench
pip install -e .
```

### Running Experiments

```bash
# Native settings (baselines)
GPU_ID=0 SEED=0 bash bash/native/run_da.sh
GPU_ID=0 SEED=0 bash bash/native/run_dg.sh
GPU_ID=0 SEED=0 bash bash/native/run_cl.sh

# Cross-setting experiments
GPU_ID=0 SEED=0 bash bash/cross/da_on_cl.sh
GPU_ID=0 SEED=0 bash bash/cross/da_on_dg.sh
GPU_ID=0 SEED=0 bash bash/cross/dg_on_cl.sh
GPU_ID=0 SEED=0 bash bash/cross/dg_on_da.sh
GPU_ID=0 SEED=0 bash bash/cross/cl_on_da.sh
GPU_ID=0 SEED=0 bash bash/cross/cl_on_dg.sh
```

---

## Evaluation Metrics

### CL Setting Metrics
- Average Accuracy (Avg. Acc)
- Forgetting
- Forward Transfer (FWT)
- Backward Transfer (BWT)

### DA Setting Metrics  
- Target Domain Accuracy
- Per-domain Accuracy Matrix
- A-Distance

### DG Setting Metrics
- Leave-one-out Accuracy
- Worst-domain Accuracy
- Average Accuracy

---

## Directory Structure

```
openworld-bench/
├── openworld/           # Main package
│   ├── methods/         # DA, DG, CL methods
│   ├── datasets/        # Dataset loaders
│   ├── protocols/       # Cross-setting evaluation
│   ├── metrics/         # Setting-specific metrics
│   └── utils/           # Utilities
├── configs/             # Configuration files
├── scripts/             # Training scripts
├── bash/                # Executable bash scripts
│   ├── native/          # Native setting runs
│   └── cross/           # Cross-setting runs
└── docs/                # Detailed documentation
```

---

## Citation

```bibtex
@misc{openworld-bench,
  title={openworld-bench: Cross-Setting Evaluation of DA, DG, and CL Methods},
  year={2024}
}
```

## Acknowledgments

Built upon:
- [SACK-CL / Mammoth](https://github.com/aimagelab/mammoth) for CL methods
- [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) for DA/DG methods
