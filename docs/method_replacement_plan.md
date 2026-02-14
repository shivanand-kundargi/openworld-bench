# Method Replacement Plan: Official Repository Ports

## Goal

Replace all 9 method implementations in `openworld-bench` with strict ports from their official repositories:
- **CL methods** → from [mammoth](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/) (local)
- **DA methods** → from [Transfer-Learning-Library (TLL)](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/Transfer-Learning-Library/) (local)
- **DG methods** → from DomainBed / official paper repos (search online)

> [!IMPORTANT]
> The openworld-bench base classes (`CLMethod`, `DAMethod`, `DGMethod`) define fixed `observe()` signatures that differ from mammoth/TLL. We port the **exact core logic** (loss formulations, buffer ops, architectural components) while wrapping them in our interface. For TLL, we **directly import** TLL modules.

---

## Phase 1: DA Methods (from TLL — direct imports)

TLL provides clean, modular loss modules that can be imported directly. We add `Transfer-Learning-Library` to `sys.path` and import loss/architecture modules.

### Shared TLL Imports

We will create a helper file `openworld/methods/da/_tll_imports.py` that handles path setup:

```python
import sys
sys.path.insert(0, '/umbc/rs/pi_gokhale/users/shivank2/shivanand/Transfer-Learning-Library')
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss, RandomizedMultiLinearMap, MultiLinearMap
from tllib.alignment.mcd import classifier_discrepancy, ImageClassifierHead
from tllib.alignment.coral import CorrelationAlignmentLoss
```

---

### 1.1 DANN — [dann.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/da/dann.py)

**Reference**: [TLL dann.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/Transfer-Learning-Library/tllib/alignment/dann.py)

| Aspect | Current (wrong) | TLL (correct) |
|---|---|---|
| GRL | Static `GradientReversalLayer(alpha=1.0)` | `WarmStartGradientReverseLayer(alpha=1, lo=0, hi=1, max_iters=1000, auto_step=True)` — alpha anneals from 0→1 |
| Discriminator | Custom 3-layer `DomainDiscriminator` with BN+ReLU | TLL's `DomainDiscriminator(in_feature, hidden_size, batch_norm=True, sigmoid=True)` — same architecture but official |
| Loss | Single `F.binary_cross_entropy(domain_pred, domain_labels)` | `0.5 * (BCE(d_s, 1, w_s) + BCE(d_t, 0, w_t))` — separate source/target losses, supports instance weighting |
| Domain accuracy | Not tracked | Tracked via `binary_accuracy` |

**Changes**: 
- Remove custom `GradientReversalFunction`, `GradientReversalLayer`, `DomainDiscriminator` classes
- Import from TLL: `WarmStartGradientReverseLayer`, `DomainDiscriminator`, `DomainAdversarialLoss`
- Use `DomainAdversarialLoss` module for domain loss computation (it wraps GRL + discriminator + BCE)
- Keep bottleneck + classifier architecture, keep `observe(x_src, y_src, x_tgt)` signature

---

### 1.2 CDAN — [cdan.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/da/cdan.py)

**Reference**: [TLL cdan.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/Transfer-Learning-Library/tllib/alignment/cdan.py)

| Aspect | Current (wrong) | TLL (correct) |
|---|---|---|
| GRL | Static `GradientReversalLayer(alpha=1.0)` (imported from dann.py) | `WarmStartGradientReverseLayer` via `ConditionalDomainAdversarialLoss` |
| Conditioning | Custom `RandomizedMultilinearMap` with `nn.Linear` (learnable weights) | TLL's `RandomizedMultiLinearMap` with fixed random projections (no grad), or `MultiLinearMap` for full outer product |
| Entropy conditioning | Not supported | Supported via `entropy_conditioning` flag |
| Loss | Manual GRL → discriminator → BCE pipeline | `ConditionalDomainAdversarialLoss(discriminator, entropy_conditioning, randomized, ...)` module |

**Changes**:
- Remove custom `RandomizedMultilinearMap` class
- Import `ConditionalDomainAdversarialLoss` from TLL (handles GRL + conditioning + discriminator + loss)
- Import `DomainDiscriminator` from TLL for the domain discriminator
- Keep bottleneck + classifier, keep `observe()` signature

---

### 1.3 MCD — [mcd.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/da/mcd.py)

**Reference**: [TLL mcd.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/Transfer-Learning-Library/tllib/alignment/mcd.py)

| Aspect | Current (wrong) | TLL (correct) |
|---|---|---|
| Discrepancy | Custom `_classifier_discrepancy` with softmax → L1 | TLL's `classifier_discrepancy(pred1, pred2)` = `mean(abs(pred1 - pred2))` — operates on raw softmax predictions |
| Classifier head | Custom 2-layer MLP (Linear→ReLU→Linear) | TLL's `ImageClassifierHead` with Dropout(0.5) + BN + ReLU (3 layers, deeper) |
| Training | Single-step (loss = cls - discrepancy) — **fundamentally broken** | 3-step alternating: (A) train all on source CE, (B) freeze G, maximize discrepancy, (C) freeze classifiers, minimize discrepancy |

> [!CAUTION]
> The current MCD implementation is a single-step version that does `cls_loss - trade_off * discrepancy`. MCD's core contribution is the **3-step alternating optimization** (Step A: classify source, Step B: maximize discrepancy on target, Step C: minimize discrepancy via generator). Without this, MCD degenerates. The replacement must implement the full 3-step procedure.

**Changes**:
- Import `classifier_discrepancy` and `ImageClassifierHead` from TLL
- Implement 3-step alternating training in `observe()`:
  1. Step A: Train backbone + both classifiers on source CE
  2. Step B: Freeze backbone, train classifiers to maximize target discrepancy
  3. Step C: Freeze classifiers, train backbone to minimize target discrepancy (num_k steps)
- Use TLL's `ImageClassifierHead` architecture for both classifiers

---

## Phase 2: CL Methods (from mammoth — port exact logic)

mammoth's CL methods are tightly coupled to `ContinualModel` base class (manages optimizer, buffer, task offsets, etc.). We **cannot import directly** — instead we port the exact loss formulations while wrapping in our `CLMethod` interface.

### Key mammoth Conventions
- `observe(inputs, labels, not_aug_inputs)` → we adapt to `observe(x, y, task_id)`  
- `self.net(inputs)` returns full logits → we use `self.backbone(x)` + `self.classifier(f)`
- `self.loss(outputs, labels)` = `CrossEntropyLoss` → we use `F.cross_entropy`
- `self.buffer` = mammoth `Buffer` → we use our `Buffer` class
- Properties: `self.current_task`, `self.n_past_classes`, `self.n_seen_classes`, `self.cpt`

---

### 2.1 iCaRL — [icarl.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/cl/icarl.py)

**Reference**: [mammoth icarl.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/models/icarl.py)

| Aspect | Current (wrong) | mammoth (correct) |
|---|---|---|
| Loss formulation | CE + separate distillation + replay as three terms | `F.binary_cross_entropy_with_logits(outputs, combined_targets)` — single BCE on one-hot + old soft targets |
| First task | CE loss | BCE with one-hot targets `self.eye[labels]` |
| Later tasks | CE + KL distillation + buffer replay | BCE with combined targets: `cat(sigmoid(old_logits[:, :past]), one_hot[:, past:])` — single loss |
| Buffer filling | Simple reservoir sampling | Herding-based: `fill_buffer(..., use_herding=True, normalize_features=True)` |
| Forward (NCM) | Simple mean-of-features per class | Averaged features + flipped features, normalized: `(mean(feats) + mean(hflip(feats))) / 2` |
| Weight decay | Not present | Custom: `self.wd() * self.args.opt_wd` added to loss (manual L2 on all params) |
| Data tracking | Not present | `self.classes_so_far` register_buffer, `self.allx/ally` for replay merging |

**Changes**:
- Replace loss with mammoth's BCE formulation:
  - Task 0: `F.binary_cross_entropy_with_logits(logits, eye[labels])`
  - Task > 0: `F.binary_cross_entropy_with_logits(logits, cat(sigmoid(old_logits[:, :past]), eye[labels][:, past:]))`
- Store old model at end_task: `self.old_net = deepcopy(backbone+classifier)`
- Implement NCM forward with averaged + flipped features
- Add manual weight decay `sum(p.pow(2).sum())` to loss
- Track `classes_so_far` for proper NCM computation
- Keep our `observe(x, y, task_id)` signature

---

### 2.2 DER — [der.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/cl/der.py)

**Reference**: [mammoth der.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/models/der.py)

| Aspect | Current (wrong) | mammoth (correct) |
|---|---|---|
| Buffer storage | Stores augmented `x` | Stores `not_aug_inputs` (non-augmented) |
| Buffer replay | `get_data(min(x.size(0), 64), return_logits=True)` retrieves x,y,logits | `get_data(minibatch_size, transform=self.transform)` — retrieves x,logits only (no labels), applies transform on retrieval |
| DER loss | Has dimension mismatch handling (min_dim) | Direct `F.mse_loss(buf_outputs, buf_logits)` — no dimension handling |
| Buffer add | `buffer.add_data(x.cpu(), y.cpu(), logits=logits.cpu())` | `buffer.add_data(examples=not_aug_inputs, logits=outputs.data)` — no labels stored for DER |

> [!NOTE]
> mammoth DER is remarkably simple: 49 lines total. The key difference is that it stores **non-augmented inputs** and applies transforms at replay time. Our version stores augmented inputs. Since we don't have `not_aug_inputs` in our pipeline (our `observe(x, y, task_id)` only gets augmented `x`), we will store the augmented inputs but note this deviation.

**Changes**:
- Simplify to match mammoth's clean logic
- Use `self.args.alpha * F.mse_loss(buf_outputs, buf_logits)` directly (no dimension handling)
- Store detached logits alongside inputs in buffer
- Remove unnecessary labels storage for DER replay (DER only uses logits, not labels)

---

### 2.3 LwF — [lwf.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/cl/lwf.py)

**Reference**: [mammoth lwf.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/models/lwf.py)

| Aspect | Current (wrong) | mammoth (correct) |
|---|---|---|
| Distillation loss | `F.kl_div(log_softmax(new/T), softmax(old/T))` — standard KD | `modified_kl_div(smooth(soft(old), T), smooth(soft(new), T))` = `-mean(sum(old_smooth * log(new_smooth)))` — custom formulation |
| `smooth()` function | Not present | `log = logits ** (1/temp); return log / sum(log)` — power-law smoothing, not softmax |
| Warm-up | Not present | `begin_task`: fine-tune only classifier on new-task data for `n_epochs` using SGD, then compute cached logits for all training data |
| Cached logits | Old model forward at train time | Pre-computed at `begin_task`, stored in `dataset.train_loader.dataset.logits` |
| Loss scope | Applied to all classes | `self.loss(outputs[:, :n_seen_classes], labels)` — loss only on seen classes |

> [!WARNING]
> mammoth's LwF `begin_task` heavily depends on the mammoth dataset interface (`dataset.train_loader.dataset.data`, `.logits`, `.extra_return_fields`). We cannot replicate the exact warm-up + cached logits mechanism with our pipeline. Instead, we will:
> 1. Port the exact `smooth()` and `modified_kl_div()` functions
> 2. Port the loss formulation (CE on seen classes + alpha * modified_kl_div on past classes)
> 3. Skip the warm-up and cached logits (compute distillation on-the-fly using `old_net`)

**Changes**:
- Add mammoth's `smooth(logits, temp, dim)` function (power-law, not softmax)
- Add mammoth's `modified_kl_div(old, new)` function
- Use `self.loss(outputs[:, :n_seen_classes], labels)` for classification
- Compute distillation on-the-fly: `modified_kl_div(smooth(soft(old_logits[:, :past]), T), smooth(soft(new_logits[:, :past]), T))`
- Store old model at end of task

---

### 2.4 CODA-Prompt — [coda_prompt.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/cl/coda_prompt.py)

**Reference**: [mammoth coda_prompt.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/models/coda_prompt.py) + [coda_prompt_utils/](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/models/coda_prompt_utils/)

> [!CAUTION]
> mammoth CODA-Prompt uses a **custom ViT backbone** (`vit_base_patch16_224`) with prompts prepended to ViT tokens at each transformer layer — this is fundamentally incompatible with generic CNN backbones (ResNet). Our current version uses a "residual connection" workaround that is **not faithful** to the paper. 
>
> **Decision needed**: Either (a) require ViT backbone for CODA-Prompt and import mammoth's `coda_prompt_utils/model.py`, or (b) keep our simplified adaptation with a clear disclaimer. I recommend option (a) with a fallback disclaimer.

**Changes**:
- Import mammoth's `coda_prompt_utils/model.py` `Model` class
- Port mammoth's exact `observe()`: `loss = CE(logits[:offset2], labels) + mu * loss_prompt.sum()`
- Port offset masking: `logits[:, :offset1] = -inf`
- Port virtual batch size support
- Port `begin_task()`: `net.task_id = task`, `net.prompt.process_task_count()`, cosine scheduler

---

### 2.5 X-DER — [xder.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/cl/xder.py)

**Reference**: [mammoth xder.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/mammoth/models/xder.py)

| Aspect | Current (wrong) | mammoth (correct) |
|---|---|---|
| SimCLR loss | Same-forward twice, custom implementation | `SupConLoss` from `utils.simclrloss`, uses actual augmentations via `gpu_augmentation` |
| Logit update | Simple EMA update at end_task | `update_logits()` with gamma-scaled transplantation + stochastic update counter |
| Buffer draws | Single buffer draw | Two separate draws (`buf_idx1` + `buf_idx2`) with deduplication |
| BN alignment | Not present | `bn_track_stats` context manager for proper BN behavior |
| Constraints | Not present | Future constraint (margin), Past constraint (softmax margin), DP loss (SPKD) |
| end_task | Simple logit re-computation | Class-balanced buffer re-balancing + herding + future-past logit transplant |

**Changes**:
- Port mammoth's full `observe()` logic:
  - `loss_stream` = CE on present head only (`outputs[:, n_past:n_seen]`, `labels - n_past`)
  - `loss_der` = alpha * MSE on buffer logits  
  - `loss_derpp` = beta * CE on buffer labels (past heads only)
  - `loss_cons` = SimCLR via SupConLoss on future heads
  - `loss_constr_futu` = margin constraint on future vs current heads
  - `loss_constr_past` = margin constraint on past vs current heads
- Port `update_logits()` with gamma-scaled transplantation
- Port `end_task()` with buffer rebalancing and future-past logit updates
- Adapt SupConLoss (port from mammoth or implement equivalent)

---

### 2.6 MEMO — [memo.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/cl/memo.py)

**Reference**: NOT in mammoth. Official: [wangkiw/ICLR23-MEMO](https://github.com/wangkiw/ICLR23-MEMO)

> [!IMPORTANT]
> MEMO is not available in mammoth. I will search the official GitHub repo and port from there. The current implementation claims to be adapted from the official repo but uses a simplified `AdaptiveExtractor` (2-layer MLP) instead of actual ResNet block splitting.

**Changes** (pending official repo review):
- Search official repo for exact `_update_representation()` and `AdaptiveNet` architecture
- Port the real block-splitting mechanism (generalized vs specialized ResNet blocks)
- Port exact auxiliary loss computation
- Port weight alignment

---

## Phase 3: DG Methods (from DomainBed / official papers)

### 3.1 IRM — [irm.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/dg/irm.py)

**Reference**: [DomainBed IRM](https://github.com/facebookresearch/DomainBed)

Current implementation **matches** the standard formulation:
- IRM penalty: `grad(CE(logits * scale, labels), [scale])^2` ✓
- Penalty annealing ✓
- Per-domain loop with averaging ✓

**Changes**: Minimal — verify penalty weight schedule matches DomainBed (currently returns 1.0 during anneal, should return `1.0` then `penalty_weight`). DomainBed uses `penalty_weight * penalty` always, no annealing phase that sets weight=1. Will update to match.

---

### 3.2 VREx — [vrex.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/dg/vrex.py)

**Reference**: [DomainBed VREx](https://github.com/facebookresearch/DomainBed)

Current implementation **matches** the standard formulation:
- Variance of per-domain losses ✓
- Penalty annealing ✓

**Changes**: Same annealing fix as IRM — match DomainBed's schedule.

---

### 3.3 CORAL — [coral.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/openworld-bench/openworld/methods/dg/coral.py)

**Reference**: [TLL coral.py](file:///umbc/rs/pi_gokhale/users/shivank2/shivanand/Transfer-Learning-Library/tllib/alignment/coral.py)

| Aspect | Current | TLL |
|---|---|---|
| Covariance loss | `(C_s - C_t).pow(2).sum() / (4*d*d)` | `(C_s - C_t).pow(2).mean()` + `(mean_s - mean_t).pow(2).mean()` (includes mean alignment) |
| Normalization | Frobenius norm / 4d² | Simple `.mean()` |
| Mean alignment | Not present | Present: `mean_diff` term |

**Changes**:
- Import TLL's `CorrelationAlignmentLoss` and use it for pairwise domain alignment
- Update `observe()` to use `CorrelationAlignmentLoss().forward(f_domain_i, f_domain_j)` for each domain pair

---

## Verification Plan

### Automated
1. **Import test**: Run `python -c "from openworld.methods.da.dann import DANN; ..."` for all methods
2. **Instantiation test**: Create each method with a dummy backbone and verify no errors
3. **Forward pass test**: Run a single `observe()` call with random data for each method
4. **Loss sanity check**: Verify losses are finite and non-zero

### Manual
1. Run a short training (5 epochs) for one method per category on OfficeHome
2. Verify loss curves look reasonable (non-trivial, not NaN)

---

## File Change Summary

| File | Action | Source |
|---|---|---|
| `methods/da/_tll_imports.py` | **NEW** | Helper for TLL path setup |
| `methods/da/dann.py` | MODIFY | Port from TLL |
| `methods/da/cdan.py` | MODIFY | Port from TLL |
| `methods/da/mcd.py` | MODIFY | Port from TLL |
| `methods/cl/icarl.py` | MODIFY | Port from mammoth |
| `methods/cl/der.py` | MODIFY | Port from mammoth |
| `methods/cl/lwf.py` | MODIFY | Port from mammoth |
| `methods/cl/coda_prompt.py` | MODIFY | Port from mammoth + coda_prompt_utils |
| `methods/cl/xder.py` | MODIFY | Port from mammoth |
| `methods/cl/memo.py` | MODIFY | Port from official GitHub |
| `methods/dg/irm.py` | MODIFY | Verify/fix against DomainBed |
| `methods/dg/vrex.py` | MODIFY | Verify/fix against DomainBed |
| `methods/dg/coral.py` | MODIFY | Port loss from TLL |
