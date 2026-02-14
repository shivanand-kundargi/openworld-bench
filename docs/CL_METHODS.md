# Continual Learning Methods

## Classic Methods

### iCaRL (Incremental Classifier and Representation Learning)
**Paper**: iCaRL: Incremental Classifier and Representation Learning (CVPR 2017)

**Key Idea**: Maintain class exemplars in memory and use nearest-class-mean classifier with knowledge distillation.

**Components**:
- Experience replay buffer
- Knowledge distillation from old model
- NCM (Nearest Class Mean) classifier

**Loss**: `L = L_task + L_distillation + L_replay`

---

### DER (Dark Experience Replay)
**Paper**: Dark Experience for General Continual Learning (NeurIPS 2020)

**Key Idea**: Store and replay not just inputs/labels, but also past model logits to better preserve knowledge.

**Components**:
- Experience replay buffer with logits
- MSE loss between current and stored logits

**Loss**: `L = L_task + α * MSE(logits_current, logits_stored)`

---

### LwF (Learning without Forgetting)
**Paper**: Learning without Forgetting (ECCV 2016)

**Key Idea**: Use knowledge distillation alone (no buffer) to prevent forgetting by matching old model outputs.

**Components**:
- Old model copy for distillation
- Temperature-scaled softmax targets

**Loss**: `L = L_task + α * KL(softmax(logits/T) || softmax(old_logits/T))`

---

## Recent Methods (To Be Added)

### CODA-Prompt (CVPR 2023)
*Implementation pending*

### X-DER (TPAMI 2023)
*Implementation pending*

### MEMO (ICLR 2024)
*Implementation pending*

---

## Implementation Files

- `openworld/methods/cl/base.py` - Base class with Buffer
- `openworld/methods/cl/icarl.py` - iCaRL implementation
- `openworld/methods/cl/der.py` - DER implementation
- `openworld/methods/cl/lwf.py` - LwF implementation

---

## CL-Specific Metrics

| Metric | Description |
|--------|-------------|
| **Avg. Accuracy** | Mean accuracy on all tasks after final task |
| **Forgetting** | Average drop in performance on old tasks |
| **BWT** | Backward Transfer - performance change on old tasks |
| **FWT** | Forward Transfer - zero-shot performance on new tasks |
