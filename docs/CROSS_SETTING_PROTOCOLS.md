# Cross-Setting Evaluation Protocols

This document describes how methods from one setting (DA, DG, CL) are evaluated under different settings.

---

## Evaluation Matrix

| Method Origin | DA Setting | DG Setting | CL Setting |
|--------------|------------|------------|------------|
| **DA Methods** | ✅ Native | Cross | Cross |
| **DG Methods** | Cross | ✅ Native | Cross |
| **CL Methods** | Cross | Cross | ✅ Native |

---

## Protocol Definitions

### DA Methods → CL Setting (`da_on_cl.sh`)

**Challenge**: DA methods expect simultaneous access to source and target data. CL setting provides sequential task data.

**Implementation**:
- Data presented as sequential tasks (Task 1 → Task 2 → ... → Task N)
- Each task contains classes never seen before
- DA method observes ONLY current task data
- No "target domain" available during training
- Domain discriminator operates within single task

**Expected Behavior**: DA methods should struggle because:
- No multi-domain signal within single task
- Domain adversarial loss becomes meaningless
- No mechanism to prevent forgetting

**Metrics**: Avg. Accuracy, Forgetting, BWT

---

### DA Methods → DG Setting (`da_on_dg.sh`)

**Challenge**: DA methods require target domain data. DG setting has NO target data during training.

**Implementation**:
- Source domains available simultaneously (like DA)
- But target domain is held out completely
- DA method trains on source domains only
- Evaluated on unseen target domain

**Expected Behavior**: DA methods should underperform because:
- Domain discriminator trained without target
- Cannot align to unknown target distribution
- Adversarial training may overfit to source

**Metrics**: Leave-one-out Accuracy, Worst-domain Accuracy

---

### DG Methods → CL Setting (`dg_on_cl.sh`)

**Challenge**: DG methods assume simultaneous multi-domain access. CL provides one task at a time.

**Implementation**:
- Data presented sequentially as tasks
- DG method processes single task at a time
- Invariance penalties applied within-task only
- No cross-task invariance signal

**Expected Behavior**: DG methods should struggle because:
- IRM/VREx penalties require multiple domains
- Single-domain penalty is trivially zero
- No mechanism to prevent forgetting

**Metrics**: Avg. Accuracy, Forgetting, BWT

---

### DG Methods → DA Setting (`dg_on_da.sh`)

**Challenge**: DG methods don't use target data. DA provides unlabeled target data.

**Implementation**:
- Source domains + unlabeled target available
- DG method may or may not use target features
- Invariance applied across source domains
- Target data available but unlabeled

**Expected Behavior**: DG methods might perform reasonably because:
- Can still learn source-invariant features
- But miss opportunity to adapt to target
- No domain alignment to target

**Metrics**: Target Accuracy, Per-domain Accuracy

---

### CL Methods → DA Setting (`cl_on_da.sh`)

**Challenge**: CL methods expect sequential tasks. DA has simultaneous source + target.

**Implementation**:
- Domains treated as sequential "tasks"
- Train on source domains sequentially
- Buffer stores samples across domains
- Finally evaluate on target domain

**Expected Behavior**: CL methods might help because:
- Buffer provides cross-domain memory
- But lacks explicit domain alignment
- May forget early source domains

**Metrics**: Target Accuracy, Per-domain Accuracy

---

### CL Methods → DG Setting (`cl_on_dg.sh`)

**Challenge**: CL methods don't optimize for generalization. DG requires generalization to unseen.

**Implementation**:
- Source domains presented sequentially
- Buffer maintains cross-domain samples
- Test on held-out (never seen) domain

**Expected Behavior**: CL methods may partially help because:
- Buffer provides multi-domain signal
- Distillation might preserve diverse features
- But no explicit invariance objective

**Metrics**: Leave-one-out Accuracy, Worst-domain Accuracy

---

## Running Experiments

### Native Settings (Baselines)
```bash
GPU_ID=0 SEED=0 bash bash/native/run_da.sh  # DA on DA
GPU_ID=0 SEED=0 bash bash/native/run_dg.sh  # DG on DG
GPU_ID=0 SEED=0 bash bash/native/run_cl.sh  # CL on CL
```

### Cross-Settings
```bash
# DA methods → other settings
GPU_ID=0 SEED=0 bash bash/cross/da_on_cl.sh
GPU_ID=0 SEED=0 bash bash/cross/da_on_dg.sh

# DG methods → other settings
GPU_ID=0 SEED=0 bash bash/cross/dg_on_cl.sh
GPU_ID=0 SEED=0 bash bash/cross/dg_on_da.sh

# CL methods → other settings
GPU_ID=0 SEED=0 bash bash/cross/cl_on_da.sh
GPU_ID=0 SEED=0 bash bash/cross/cl_on_dg.sh
```

---

## Expected Results Summary

| Method → Setting | Expected Outcome |
|-----------------|------------------|
| DA → CL | **Very Poor** (no forgetting prevention) |
| DA → DG | **Poor** (no target, adversarial overfits) |
| DG → CL | **Very Poor** (no invariance signal) |
| DG → DA | **Moderate** (invariant features, no adaptation) |
| CL → DA | **Moderate** (buffer helps, no alignment) |
| CL → DG | **Moderate** (buffer helps, no invariance) |

These results support the hypothesis that methods designed for single settings fail in others, motivating the need for unified approaches.
