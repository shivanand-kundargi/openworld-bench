# Settings Explanation — openworld-bench

Every detail below is taken directly from `scripts/train.py` (909 lines) and `openworld/datasets/domain_datasets.py` (243 lines). Nothing is assumed or extrapolated.

---

## Table of Contents

1. [Global Configuration](#1-global-configuration)
2. [Data Loading & Transforms](#2-data-loading--transforms)
3. [Method Instantiation](#3-method-instantiation)
4. [Evaluation Functions](#4-evaluation-functions)
5. [Native Settings (3)](#5-native-settings)
6. [Cross Settings (6)](#6-cross-settings)
7. [Output & Results](#7-output--results)

---

## 1. Global Configuration

All experiments are launched via `scripts/train.py` with the following arguments (lines 43–84):

| Argument | Type | Default | Choices / Notes |
|---|---|---|---|
| `--method` | str | *required* | `dann`, `cdan`, `mcd`, `irm`, `vrex`, `coral`, `icarl`, `der`, `lwf` |
| `--setting` | str | *required* | `da`, `dg`, `cl` |
| `--dataset` | str | *required* | `domainnet`, `office_home`, `imagenet_r`, `cub200`, `stanford_cars`, `fgvc_aircraft`, `inaturalist`, `cifar100` |
| `--data_root` | str | `./data` | Base directory for CL datasets |
| `--source_domains` | str (nargs+) | `None` | DA/DG only. If None, defaults are used (see below) |
| `--target_domain` | str | `None` | DA/DG only. If None, defaults are used |
| `--n_tasks` | int | `10` | CL only. Number of sequential tasks |
| `--buffer_size` | int | `500` | CL methods only (iCaRL, DER) |
| `--backbone` | str | `resnet50` | Backbone architecture |
| `--epochs` | int | `50` | Training epochs per task/setting |
| `--batch_size` | int | `32` | Batch size for all loaders |
| `--lr` | float | `0.001` | Learning rate |
| `--weight_decay` | float | `1e-4` | Weight decay |
| `--trade_off` | float | `1.0` | Method-specific penalty weight (DA: adversarial, DG: invariance, CORAL: coral_weight) |
| `--alpha` | float | `0.5` | Used by DER (distillation weight) and LwF (distillation weight) |
| `--seed` | int | `0` | Random seed |
| `--gpu` | int | `0` | GPU device ID |
| `--output_dir` | str | `./outputs` | Output directory |

### Optimizer (lines 840–841, 882–883)
All methods use **SGD** with:
- `lr=args.lr` (default 0.001)
- `momentum=0.9`
- `weight_decay=args.weight_decay` (default 1e-4)

### Cross-setting detection (lines 807–808)
```python
method_origin = get_method_origin(args.method)  # 'da', 'dg', or 'cl'
is_cross = method_origin != args.setting
```

### Method origin mapping (lines 232–240)
- **DA methods**: `dann`, `cdan`, `mcd`, `toalign`, `pmtrans`
- **DG methods**: `irm`, `vrex`, `coral`, `swad`, `miro`, `eoa`
- **CL methods**: `icarl`, `der`, `lwf`, `coda_prompt`, `xder`, `memo`

### Default domain splits (lines 858–870)
If `--source_domains` and `--target_domain` not provided:
- **Office-Home**: source = `['Art', 'Clipart', 'Product']`, target = `'Real_World'`
- **DomainNet**: source = `['clipart', 'infograph', 'painting', 'quickdraw', 'real']`, target = `'sketch'`

---

## 2. Data Loading & Transforms

### CL Datasets (lines 91–105, 155–200)

**CIFAR-100** — `get_cl_dataset()` line 157:
- `datasets.CIFAR100(data_root, train=True/False, download=True)`
- 100 classes, split into `n_tasks` tasks (default 10 → 10 classes/task)
- Train transform: `RandomCrop(32, padding=4)` → `RandomHorizontalFlip()` → `Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))`
- Test transform: `Normalize` only (same means/stds)

**Other CL datasets** (ImageNet-R, CUB-200, etc.) — lines 164–200:
- Auto-discovers paths via `_find_dataset_path()` which searches `data_root` and a fallback SACK-CL directory
- If train/test split directories exist, uses `ImageFolder` on each
- Otherwise: 80/20 random split with `np.random.RandomState(42)`
- If `num_classes % n_tasks != 0`, auto-adjusts n_tasks trying `[10, 20, 5, 25, 4, 8]`
- Transform: `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip()` → `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

**CL task splitting** — `split_dataset_by_classes()` lines 203–225:
- Classes `[t*cpt, (t+1)*cpt)` assigned to task `t`, where `cpt = num_classes // n_tasks`
- Each task is a `torch.utils.data.Subset` of the original dataset

### DA/DG Datasets (domain_datasets.py)

**Data loaders** — `MultiDomainDataset` lines 71–141:
- `get_source_loaders(batch_size)` → `Dict[str, DataLoader]`: one loader per source domain
  - `shuffle=True`, `drop_last=True`, `num_workers=4`, `pin_memory=True`
- `get_combined_source_loader(batch_size)` → single loader over `ConcatDataset` of all sources
  - `shuffle=True`, `drop_last=True`, `num_workers=4`, `pin_memory=True`
- `get_target_loader(batch_size)` → target domain loader
  - `shuffle=False`, `num_workers=4`, `pin_memory=True`
  - Target uses `test_transform` (applied via `test_transform or train_transform`)

**Office-Home** — lines 146–167:
- 4 domains: `Art`, `Clipart`, `Product`, `Real_World`
- 65 classes
- Auto-discovered from hardcoded paths

**DomainNet** — lines 170–192:
- 6 domains: `clipart`, `infograph`, `painting`, `quickdraw`, `real`, `sketch`
- 345 classes
- Auto-discovered from hardcoded paths

**DA/DG transforms** — `get_da_transforms()` lines 225–242:
- Train: `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip()` → `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- Test: `Resize(256)` → `CenterCrop(224)` → `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

---

## 3. Method Instantiation

All methods are created via `get_method()` (lines 243–284). Each receives:
- `backbone`: ResNet50 (from `get_backbone()`)
- `num_classes`: dataset-dependent
- `device`: CUDA device
- `feature_dim`: backbone output dimension

| Method | Class | Key hyperparameters |
|---|---|---|
| DANN | `DANN(backbone, num_classes, device, feature_dim, trade_off=args.trade_off)` | `trade_off` (default 1.0) — weight of adversarial loss |
| CDAN | `CDAN(backbone, num_classes, device, feature_dim, trade_off=args.trade_off)` | `trade_off` (default 1.0) — weight of conditional adversarial loss |
| MCD | `MCD(backbone, num_classes, device, feature_dim, trade_off=args.trade_off)` | `trade_off` (default 1.0) — weight of discrepancy loss |
| IRM | `IRM(backbone, num_classes, device, feature_dim, penalty_weight=args.trade_off)` | `trade_off` (default 1.0) — invariant risk penalty weight |
| VREx | `VREx(backbone, num_classes, device, feature_dim, penalty_weight=args.trade_off)` | `trade_off` (default 1.0) — variance penalty weight |
| CORAL | `CORAL(backbone, num_classes, device, feature_dim, coral_weight=args.trade_off)` | `trade_off` (default 1.0) — coral alignment weight |
| iCaRL | `ICarl(backbone, num_classes, device, feature_dim, buffer_size=args.buffer_size)` | `buffer_size` (default 500) |
| DER | `DER(backbone, num_classes, device, feature_dim, buffer_size=args.buffer_size, alpha=args.alpha)` | `buffer_size` (default 500), `alpha` (default 0.5) — distillation weight |
| LwF | `LwF(backbone, num_classes, device, feature_dim, alpha=args.alpha)` | `alpha` (default 0.5) — distillation weight |

---

## 4. Evaluation Functions

### `evaluate(method, loader, device)` — lines 291–303
Standard classification accuracy. Calls `method.forward(x)`, takes `argmax`, compares to labels. Uses **all logits** (no masking).

### `evaluate_cil(method, loader, device, n_seen_classes)` — lines 306–318
**Class-Incremental Learning** evaluation. Masks logits to first `n_seen_classes` via `logits[:, :n_seen_classes]`. Model must discriminate among ALL classes seen so far without being told which task a sample belongs to.

### `evaluate_til(method, loader, device, task_class_start, task_class_end)` — lines 321–340
**Task-Incremental Learning** evaluation. Masks logits to `logits[:, task_class_start:task_class_end]`. Labels remapped to task-local: `local_y = y - task_class_start`. Model is "told" which task the sample belongs to and only classifies within that task's class range.

---

## 5. Native Settings

### 5.1 DA → DA (Native DA)

**Location**: `train_da_setting()`, `method_origin == 'da'` branch, lines 569–599

**Data loaders used**:
- `source_loader` = `domain_data.get_combined_source_loader(args.batch_size)` — all source domains combined
- `target_loader` = `domain_data.get_target_loader(args.batch_size)` — target domain

**Training loop**:
```
for epoch in range(args.epochs):           # 50 epochs default
    target_iter = iter(target_loader)
    for src_x, src_y in source_loader:     # iterate over combined source
        src_x, src_y → device
        tgt_x, _ = next(target_iter)       # cycle target loader on StopIteration
        tgt_x → device
        optimizer.zero_grad()
        losses = method.observe(src_x, src_y, tgt_x)    # DA native call
        optimizer.step()
```

**observe() signature**: `method.observe(x_source, y_source, x_target)` — source is labeled, target is unlabeled (labels discarded with `_`).

**Target cycling**: When `target_iter` is exhausted, it's re-initialized with `iter(target_loader)` and continues (line 584).

**Per-epoch evaluation**: Both `source_loader` and `target_loader` evaluated via `evaluate()` (lines 596–597).

**Reported metrics**: `source_accuracy`, `target_accuracy`, `per_domain_accuracy` (per source domain + target).

---

### 5.2 DG → DG (Native DG)

**Location**: `train_dg_setting()`, `method_origin == 'dg'` branch, lines 644–677

**Data loaders used**:
- `source_domain_loaders` = `domain_data.get_source_loaders(args.batch_size)` — one `DataLoader` per source domain
- `target_loader` = `domain_data.get_target_loader(args.batch_size)` — for eval only, **never used in training**

**Training loop**:
```
domain_names = list(source_domain_loaders.keys())

for epoch in range(args.epochs):           # 50 epochs default
    domain_iters = {d: iter(l) for d, l in source_domain_loaders.items()}
    min_len = min(len(l) for l in source_domain_loaders.values())

    for _ in range(min_len):               # iterate shortest domain loader
        x_list, y_list = [], []
        for dname in domain_names:
            x, y = next(domain_iters[dname])     # one batch per domain
            x_list.append(x.to(device))
            y_list.append(y.to(device))
        optimizer.zero_grad()
        losses = method.observe(x_list, y_list)  # DG native call
        optimizer.step()
```

**observe() signature**: `method.observe([x_d1, x_d2, ...], [y_d1, y_d2, ...])` — lists of per-domain tensors.

**Iteration length**: `min_len = min(len(loader) for each domain)` — stops at shortest domain. Domain iterators are cycled on `StopIteration` (lines 660–662).

**Target**: Completely held out during training. Only used in per-epoch evaluation (line 675).

**Reported metrics**: `source_accuracy`, `target_accuracy`, `per_domain_accuracy` (per source + target with `(held-out)` suffix).

---

### 5.3 CL → CL (Native CL)

**Location**: `train_cl_setting()`, `method_origin == 'cl'` branch, lines 347–453

**Data loading**:
- CL dataset loaded via `get_cl_dataset()` → `(train_set, test_set, num_classes)`
- Split into tasks: `split_dataset_by_classes(dataset, cpt, n_tasks)` where `cpt = num_classes // n_tasks`
- Each task is a `Subset` containing classes `[t*cpt, (t+1)*cpt)`

**Training loop**:
```
for t in range(n_tasks):                   # e.g. 10 tasks for CIFAR-100
    method.begin_task(t, cpt)              # if method has begin_task

    train_loader = DataLoader(train_tasks[t], batch_size=32,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    for epoch in range(args.epochs):       # 50 epochs per task
        for x, y in train_loader:
            optimizer.zero_grad()
            losses = method.observe(x, y, t)   # CL native call
            optimizer.step()

    method.end_task(t)                     # if method has end_task

    # Evaluate on all seen tasks
    for et in range(t + 1):
        cil_acc = evaluate_cil(method, test_loader[et], device, n_seen)
        til_acc = evaluate_til(method, test_loader[et], device, et*cpt, (et+1)*cpt)
```

**observe() signature**: `method.observe(x, y, task_id)` — task_id is an integer `[0, n_tasks)`.

**begin_task / end_task**: Called if the method has these attributes (iCaRL, DER, LwF all do). `begin_task(t, cpt)` receives task index and classes per task.

**Evaluation**: Both CIL and TIL accuracy tracked via two separate `CLMetrics` instances. Evaluated after each task completes, on all seen tasks `[0, ..., t]`.

**Reported metrics**: `cil_metrics` (avg_accuracy, forgetting, bwt), `til_metrics` (same), `til_cil_gap`, `cil_accuracy_matrix`, `til_accuracy_matrix`.

---

## 6. Cross Settings

### 6.1 DA → CL (DA methods in CL setting)

**Location**: `train_cl_setting()`, `method_origin == 'da'` branch, line 390–392

**Setting**: CL (sequential tasks). Data loaded and split exactly as native CL.

**observe() call**:
```python
losses = method.observe(x, y, x)   # same batch x passed as both source and target
```

**What happens**: DA methods (DANN, CDAN, MCD) expect `observe(x_source, y_source, x_target)`. Since CL has no target domain, the **same batch `x`** is passed as both source and unlabeled target. The domain discriminator receives identical source and target distributions:
- **DANN**: discriminator tries to distinguish `x` from `x` → gradient ≈ 0
- **CDAN**: same but conditional on classifier output
- **MCD**: discrepancy between two classifiers on same data ≈ 0

The adversarial/discrepancy component produces **no useful gradient**, so the method degrades to cross-entropy only (ERM).

**Sequential structure**: Yes — the outer loop iterates over `n_tasks` tasks. But DA has no `begin_task`/`end_task` (line 369: `hasattr(method, 'begin_task')` returns False for DA methods, so it's skipped). There is no replay buffer, no distillation, no anti-forgetting mechanism.

**Evaluation**: Same as native CL — CIL and TIL accuracy after each task.

**Expected result**: Catastrophic forgetting. Method has no mechanism to prevent it.

---

### 6.2 DA → DG (DA methods in DG setting)

**Location**: `train_dg_setting()`, `method_origin == 'da'` branch, lines 679–728

**Setting**: DG (target held out). Data loaded via `domain_datasets`.

**observe() call**:
```python
# Rotate source domain pairs each step
src_idx = step % len(domain_names)
tgt_idx = (step + 1) % len(domain_names)
src_dname = domain_names[src_idx]
tgt_dname = domain_names[tgt_idx]

src_x, src_y = next(domain_iters[src_dname])
tgt_x, _     = next(domain_iters[tgt_dname])

losses = method.observe(src_x, src_y, tgt_x)
```

**What happens**: DA methods need a "source" and "target" for adversarial alignment. Since the real target is held out (DG constraint), the code **cycles through pairs of source domains**:
- Step 0: source = `domain_names[0]` (e.g., Art), target = `domain_names[1]` (e.g., Clipart)
- Step 1: source = `domain_names[1]` (Clipart), target = `domain_names[2]` (Product)
- Step 2: source = `domain_names[2]` (Product), target = `domain_names[0]` (Art)
- ... and so on, using modular arithmetic

The domain discriminator receives **genuinely different domains**, so the adversarial gradient is meaningful. DANN/CDAN/MCD learn domain-invariant features across source domains.

**Target data**: **Never used in training.** Only used for per-epoch evaluation (line 726). The held-out target constraint of DG is fully respected.

**Iteration length**: `min_len = min(len(l) for l in source_domain_loaders.values())` — same as native DG.

**Reported metrics**: source_accuracy, target_accuracy, per_domain_accuracy.

---

### 6.3 DG → DA (DG methods in DA setting)

**Location**: `train_da_setting()`, `method_origin == 'dg'` branch, lines 481–519

**Setting**: DA (target available). Data loaded via `domain_datasets`.

**observe() call**:
```python
domain_iters = {d: iter(l) for d, l in source_domain_loaders.items()}
for _ in range(min_len):
    x_list, y_list = [], []
    for dname in domain_names:
        x, y = next(domain_iters[dname])
        x_list.append(x.to(device))
        y_list.append(y.to(device))
    losses = method.observe(x_list, y_list)    # same as native DG call
```

**What happens**: DG methods receive per-domain source batches in the exact same format as native DG. The **target data is loaded** (`target_loader` created at line 475) but **never passed** to the DG method because DG methods' `observe()` does not accept target data.

**Key warning** (line 487): The code explicitly logs:
```
WARNING: DG→DA is equivalent to native DG (target unused)
```

This cross-setting produces the **same model** as native DG. The invariance penalties (IRM, VREx, CORAL) operate on the same source domains in the same way. Target data exists but DG has no mechanism to use it.

**Target data in evaluation**: Yes — per-epoch eval includes both `source_loader` and `target_loader` (lines 516–517). So the evaluation captures target accuracy, but the model itself is identical to native DG.

**Research value**: Demonstrates that DG methods **cannot leverage target data** even when available. The target accuracy reported is the same as what native DG would achieve.

---

### 6.4 DG → CL (DG methods in CL setting)

**Location**: `train_cl_setting()`, `method_origin == 'dg'` branch, lines 393–395

**Setting**: CL (sequential tasks). Data loaded and split exactly as native CL.

**observe() call**:
```python
losses = method.observe([x], [y])   # single-element list
```

**What happens**: DG methods expect `observe([x_d1, x_d2, ...], [y_d1, y_d2, ...])` — a list of per-domain batches. In CL, each task is a single domain (one class group), so the batch is wrapped in a **single-element list**. The invariance penalties compute:
- **IRM**: gradient penalty on one environment → just a regularization on one loss, no cross-environment invariance
- **VREx**: variance across `[loss_1]` → `Var([x]) = 0` → penalty is **always zero**
- **CORAL**: covariance alignment of one domain with itself → **always zero**

All penalties vanish, so the method degrades to cross-entropy only (ERM).

**Sequential structure**: Yes — iterates over tasks. But DG has no `begin_task`/`end_task` (skipped). No buffer, no distillation.

**Evaluation**: CIL and TIL accuracy after each task.

**Expected result**: Catastrophic forgetting (identical to DA→CL since both degrade to ERM).

---

### 6.5 CL → DA (CL methods in DA setting)

**Location**: `train_da_setting()`, `method_origin == 'cl'` branch, lines 521–567

**Setting**: DA (target available). Data loaded via `domain_datasets`.

**Training flow**:
```python
domain_names = list(source_domain_loaders.keys())  # e.g. ['Art', 'Clipart', 'Product']

for task_id, dname in enumerate(domain_names):     # domains as sequential tasks
    method.begin_task(task_id, 0)                    # 0 = not class-incremental

    domain_loader = source_domain_loaders[dname]

    for epoch in range(args.epochs):               # 50 epochs per domain
        for x, y in domain_loader:
            optimizer.zero_grad()
            losses = method.observe(x, y, task_id)  # CL native call
            optimizer.step()

    method.end_task(task_id)

    # Evaluate on all seen domains + target
    for prev_name in domain_names[:task_id+1]:
        acc = evaluate(method, source_domain_loaders[prev_name], device)
    tgt_acc = evaluate(method, target_loader, device)
```

**observe() call**: `method.observe(x, y, task_id)` — standard CL call. Each source domain becomes a sequential "task":
- Task 0: Art domain
- Task 1: Clipart domain
- Task 2: Product domain

**begin_task / end_task**: Called with `begin_task(task_id, 0)`. The second argument `0` indicates this is not class-incremental (all domains share the same label space). CL methods activate their task transition logic:
- **iCaRL**: updates exemplar memory with current domain samples
- **DER**: stores logits and samples in buffer
- **LwF**: creates teacher snapshot for distillation

**Buffer/distillation**: ✅ Active. When training on Clipart (task 1), the buffer contains Art samples. When training on Product (task 2), the buffer contains Art + Clipart samples. Distillation prevents forgetting earlier domains.

**Target data**: **Not used in training.** CL observe() doesn't accept target data. Target is only used for evaluation after each domain-task (line 566).

**Evaluation**: After each domain-task, evaluates `evaluate()` on all previously seen domains + target. This is standard domain accuracy (not CIL/TIL since domains share the same label space).

**Reported metrics**: source_accuracy, target_accuracy, per_domain_accuracy.

---

### 6.6 CL → DG (CL methods in DG setting)

**Location**: `train_dg_setting()`, `method_origin == 'cl'` branch, lines 730–776

**Setting**: DG (target held out). Data loaded via `domain_datasets`.

**Training flow**: **Identical structure to CL→DA** (Section 6.5), with the only difference being that the target domain is conceptually "held out":

```python
for task_id, dname in enumerate(domain_names):     # domains as sequential tasks
    method.begin_task(task_id, 0)
    domain_loader = source_domain_loaders[dname]

    for epoch in range(args.epochs):
        for x, y in domain_loader:
            optimizer.zero_grad()
            losses = method.observe(x, y, task_id)
            optimizer.step()

    method.end_task(task_id)

    # Evaluate on seen domains + held-out target
    for prev_name in domain_names[:task_id+1]:
        acc = evaluate(method, source_domain_loaders[prev_name], device)
    tgt_acc = evaluate(method, target_loader, device)
```

**observe() call**: `method.observe(x, y, task_id)` — identical to CL→DA.

**Buffer/distillation**: ✅ Active — same as CL→DA. Buffer accumulates cross-domain samples.

**Target data**: **Not used in training.** Held out for evaluation only. The evaluation log line includes `"Target-held-out"` (line 776).

**Difference from CL→DA**: In CL→DA, target data is available but CL doesn't use it (identical situation). The only difference is the conceptual framing and the log label. The trained models should be **identical** if the source domains and number of epochs are the same.

**Reported metrics**: source_accuracy, target_accuracy, per_domain_accuracy (target key includes `(held-out)` suffix).

---

## 7. Output & Results

### Output directory naming (lines 811–814)
```python
# Native: {method}_{setting}_{dataset}_seed{seed}
# Cross:  cross_{method_origin}_on_{setting}_{method}_{setting}_{dataset}_seed{seed}
```
Example: `cross_cl_on_da_icarl_da_office_home_seed0/`

### Config file (lines 898–904)
Saved as `config.json` containing all `args`, plus:
- `method_origin`: `'da'`, `'dg'`, or `'cl'`
- `is_cross_setting`: `true` or `false`
- `num_classes`: integer

### Results files

**CL setting** (`results.json`):
```json
{
  "cil_metrics": {"avg_accuracy": ..., "forgetting": ..., "bwt": ...},
  "til_metrics": {"avg_accuracy": ..., "forgetting": ..., "bwt": ...},
  "til_cil_gap": ...,
  "cil_accuracy_matrix": [[...], ...],
  "til_accuracy_matrix": [[...], ...]
}
```

**DA setting** (`results.json`):
```json
{
  "source_accuracy": ...,
  "target_accuracy": ...,
  "per_domain_accuracy": {"Art": ..., "Clipart": ..., "Product": ..., "Real_World": ...},
  "source_domains": ["Art", "Clipart", "Product"],
  "target_domain": "Real_World"
}
```

**DG setting** (`results.json`):
```json
{
  "source_accuracy": ...,
  "target_accuracy": ...,
  "per_domain_accuracy": {"Art": ..., "Clipart": ..., "Product": ..., "Real_World(held-out)": ...},
  "source_domains": ["Art", "Clipart", "Product"],
  "target_domain": "Real_World"
}
```

---

## Summary Matrix

How each method's `observe()` is called in each setting:

|  | **DA setting** | **DG setting** | **CL setting** |
|---|---|---|---|
| **DA methods** | `observe(src_x, src_y, tgt_x)` — real target | `observe(src_x, src_y, tgt_x)` — source domain pairs cycled, no real target | `observe(x, y, x)` — self-as-target, adversarial = 0 |
| **DG methods** | `observe([x_d1,...], [y_d1,...])` — per-domain, target ignored (= native DG) | `observe([x_d1,...], [y_d1,...])` — per-domain source batches | `observe([x], [y])` — single domain, penalties = 0 |
| **CL methods** | `observe(x, y, task_id)` — domains as sequential tasks, buffer active | `observe(x, y, task_id)` — domains as sequential tasks, buffer active | `observe(x, y, task_id)` — native sequential tasks |

### Key observations from the code:

1. **No method code is ever modified.** All cross-setting adaptation is done in the training loop (what data is fed to `observe()`).
2. **DG→DA = native DG.** The code logs a warning about this.
3. **CL→DA and CL→DG are structurally identical** (both present domains as sequential tasks). The only difference is log labels and per_domain_accuracy key naming.
4. **DA→CL and DG→CL both degrade to ERM** because their method-specific components receive degenerate inputs (self-as-target / single-domain list).
5. **DA→DG is the only cross-setting where the method's core mechanism genuinely activates** on non-trivial data (source domain pairs).
