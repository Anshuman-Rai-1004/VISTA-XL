# VISTA-XL Implementation Plan

> **Stage:** Planning only — no files modified, no packages installed, no training triggered.
> **Reference notebook:** [unet-project.ipynb](file:///c:/Users/LENOVO/Desktop/My%20Projects/VISTA-XL/unet-project.ipynb) (preserved unchanged, read-only reference)
> **Date:** 2026-10-14

---

## ⚠️ Open Questions (Must Resolve Before Phase 2)

The following decisions have architectural or dataset-level consequences. They are flagged explicitly and **require your approval** before implementation begins.

| # | Question | Why it matters |
|---|---|---|
| Q1 | Is the ISIC 2017 dataset already downloaded locally, or does it need to be acquired? | Determines data pipeline paths |
| Q2 | Do you want a **train / val / test** 3-way split, or use the official ISIC 2017 test set? | The official test set has ground-truth masks for ISIC 2017 — if available, it is the right choice |
| Q3 | Which GPU is available locally? (name, VRAM in GB) | Determines max safe batch size and whether full Swin-B/Swin-L variants are feasible |
| Q4 | Is the xLSTM component to be implemented from scratch, or should we use an existing library (e.g., `xlstm` PyPI package from the original authors)? | Implementation from scratch is expensive; using the library is faster but adds a dependency |
| Q5 | What is the target image resolution — 128×128 (current) or the standard 224×224 / 256×256 used by Swin-UNet papers? | Swin-UNet was designed for 224×224; using 128×128 may require architecture changes and makes fair comparison harder |
| Q6 | Should VISTA-XL be a **publication-quality** experiment (requiring statistical testing, multiple runs, fixed seeds) or a **research prototype** (single run, proof-of-concept)? | Determines how much infrastructure to build |

---

## Part 0 — Design Principles

1. [unet-project.ipynb](file:///c:/Users/LENOVO/Desktop/My%20Projects/VISTA-XL/unet-project.ipynb) is **never modified**. It serves as the original reference only.
2. All new code lives in a clean `VISTA-XL/` directory with a modular Python package structure.
3. Experiments are **sequential** and **staged**: each baseline must be verified before the next is built.
4. No results are invented. Every metric reported must come from actual executed code.
5. Claims of novelty require more than an attention block added to an existing architecture. The xLSTM integration must be meaningful and architecturally justified.

---

## Part 1 — Project Structure

```
VISTA-XL/
│
├── unet-project.ipynb          ← ORIGINAL, READ-ONLY, NEVER MODIFIED
│
├── README.md                   ← Project overview, setup, how to reproduce
├── requirements.txt            ← Pinned dependency list
├── .gitignore
│
├── data/
│   ├── isic2017/               ← Raw dataset (symlink or copy)
│   │   ├── training_data/      ← image.png + mask.png per sample
│   │   ├── validation_data/
│   │   └── test_data/          ← If official test masks are available
│   └── README.md               ← How to obtain and organize the dataset
│
├── configs/
│   ├── unet_baseline.yaml      ← Hyperparameters for U-Net experiment
│   ├── swin_unet.yaml          ← Hyperparameters for Swin-UNet experiment
│   └── vista_xl.yaml           ← Hyperparameters for VISTA-XL experiment
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              ← Dataset class, data loaders, augmentation
│   ├── transforms.py           ← All albumentations pipelines
│   ├── metrics.py              ← Dice, IoU, Precision, Recall, F1
│   ├── losses.py               ← BCEDiceLoss, FocalTverskyLoss
│   ├── trainer.py              ← Training loop, validation loop, checkpointing
│   ├── utils.py                ← Seeding, hardware check, logging helpers
│   │
│   └── models/
│       ├── __init__.py
│       ├── unet.py             ← Clean U-Net (reused + fixed from notebook)
│       ├── swin_unet.py        ← Swin-UNet implementation
│       ├── xlstm_block.py      ← xLSTM components
│       └── vista_xl.py         ← VISTA-XL = Swin-UNet + xLSTM integration
│
├── scripts/
│   ├── check_environment.py    ← Hardware + CUDA + dataset path verification
│   ├── train.py                ← Entry point: python train.py --config configs/unet_baseline.yaml
│   └── evaluate.py             ← Load checkpoint, run on test set, print metrics table
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   ← EDA: class balance, image stats, example pairs
│   └── 02_results_analysis.ipynb   ← Plot loss curves, visualize predictions
│
└── checkpoints/
    ├── unet/
    ├── swin_unet/
    └── vista_xl/
```

---

## Part 2 — Dataset & Preprocessing Specification

### 2.1 Dataset Organization

The ISIC 2017 dataset as used in the notebook has this structure:

```
training_data/
  <sample_id>/
    image.png    ← RGB dermoscopy image (variable original resolution)
    mask.png     ← Binary mask (255 = lesion, 0 = background), grayscale PNG
```

> **Uncertainty flag:** The notebook's `Utils.prepare_training_data()` suggests the raw ISIC data needs pre-processing to assemble per-sample directories. The already-processed Kaggle version (per-sample `image.png`+`mask.png`) is assumed. **Confirm this structure is intact before proceeding.**

### 2.2 Data Splitting

The notebook provides a `training_data/` and `validation_data/` split of unknown ratio. The recommended approach:

**Option A (preferred if available):** Use the official ISIC 2017 three-way split:
- Training: 2,000 images
- Validation: 150 images
- Test: 600 images

**Option B (fallback):** Use all available images and split programmatically:
```
Total images → 70% train / 15% val / 15% test
Stratified if class balance metrics are available
Fixed random seed for reproducibility
```

> **⚠️ Q2 applies here.** Confirm which split strategy to use.

### 2.3 Correct Image/Mask Tensor Shapes

| Item | Shape | dtype | Range after preprocessing |
|---|---|---|---|
| Input image | `(B, 3, H, W)` | `float32` | `[0.0, 1.0]` after ImageNet norm |
| Binary mask | `(B, 1, H, W)` | `float32` | `{0.0, 1.0}` (thresholded, not normalized to [0,1] by dividing 255 and leaving floats in between) |
| Model output (logits) | `(B, 1, H, W)` | `float32` | Unbounded (sigmoid applied only at inference/metric time) |

> **Bug fix from notebook:** The notebook's `Dataset_Class` divides mask by 255.0 — this is correct for producing `{0.0, 1.0}` values **only if** the mask is a clean binary PNG with values strictly 0 or 255. If intermediate values exist (e.g., JPEG artifacts), a threshold (`mask > 0.5`) must be applied after division.

### 2.4 Image Resolution

| Experiment | Resolution | Reason |
|---|---|---|
| U-Net baseline | 256×256 | Standard for ISIC 2017 literature; 128×128 used in notebook is below standard |
| Swin-UNet | 224×224 or 256×256 | Original Swin-UNet paper uses 224×224; Swin Transformer patch embedding requires `H % window_size == 0` |
| VISTA-XL | Same as Swin-UNet | For fair comparison |

> **⚠️ Q5 applies here.** The resolution must be confirmed before writing any model code, as Swin window sizes depend on it.

### 2.5 Normalization

Use **ImageNet channel statistics** rather than the notebook's `mean=[0,0,0], std=[1,1,1]` (no-op):

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**Justification:** Swin Transformer backbones are typically pretrained on ImageNet. Using ImageNet normalization allows future use of pretrained weights. The U-Net baseline should also use the same normalization for fair comparison.

> **Alternative:** Compute the mean/std from the ISIC 2017 training set itself and use dataset-specific normalization. This is more rigorous but requires one pass over the data before training. Flag this as a decision point.

### 2.6 Augmentation Pipeline (Training)

```python
# Training augmentation
A.Compose([
    A.Resize(height=H, width=W),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.ElasticTransform(p=0.2),   # important for medical images
    A.Normalize(mean=..., std=...),
    ToTensorV2(),
])

# Validation / Test augmentation (no random transforms)
A.Compose([
    A.Resize(height=H, width=W),
    A.Normalize(mean=..., std=...),
    ToTensorV2(),
])
```

The same albumentations transform must be applied **jointly** to image and mask (already done correctly in the notebook via `transform(image=..., mask=...)`).

---

## Part 3 — Loss Function Specification

### 3.1 Recommended Loss: BCE + Dice Combined

```
L_total = α * L_BCE + (1 - α) * L_Dice
```

Where:
- `L_BCE = BCEWithLogitsLoss(pred_logits, mask)` — numerically stable
- `L_Dice = 1 - (2 * TP + ε) / (2 * TP + FP + FN + ε)` — computed on sigmoid-activated predictions
- `α = 0.5` (equal weight; tunable)

**Why not BCE alone (as in notebook):** The ISIC dataset has significant class imbalance (background >> lesion area in many images). Pure BCE penalizes both classes equally. Dice loss directly optimizes the overlap metric used for evaluation.

**Alternative:** Focal Tversky Loss for severe imbalance cases. Flag this as an option to switch to if the Dice score plateaus early.

### 3.2 Implementation Note

The `get_DC` function in the notebook implements Dice as:
```python
DC = float(2 * Inter) / float(Inter + Union)
```
This is **not the standard Dice**. The correct formula is:
```python
Dice = 2 * TP / (2*TP + FP + FN)
```
The new `metrics.py` will implement this correctly.

---

## Part 4 — Metrics Specification

All metrics computed on **thresholded predictions** (`sigmoid(logit) > 0.5`):

| Metric | Formula | Notes |
|---|---|---|
| **Dice Coefficient** | `2·TP / (2·TP + FP + FN)` | Primary metric for segmentation |
| **IoU / Jaccard** | `TP / (TP + FP + FN)` | Standard secondary metric |
| **Sensitivity / Recall** | `TP / (TP + FN)` | Clinically important for lesion detection |
| **Specificity** | `TN / (TN + FP)` | |
| **Precision** | `TP / (TP + FP)` | |
| **F1** | `2·Precision·Recall / (Precision + Recall)` | Equivalent to Dice for binary |
| **Pixel Accuracy** | `(TP + TN) / Total` | Reported but not primary (biased by imbalance) |

Metrics are computed **per batch** and averaged over the validation/test set.

---

## Part 5 — Training, Validation and Checkpointing Protocol

### 5.1 Training Loop (per experiment)

```
for epoch in 1..N:
    model.train()
    for batch in train_loader:
        forward → loss → backward → optimizer.step()
        accumulate train_loss

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            forward → compute val_loss, Dice, IoU
            accumulate val metrics

    log_epoch_summary(epoch, train_loss, val_loss, val_dice, val_iou)
    lr_scheduler.step(val_dice)   # e.g. ReduceLROnPlateau

    if val_dice > best_val_dice:
        save_checkpoint(model, optimizer, epoch, val_dice)
        best_val_dice = val_dice
```

### 5.2 Checkpointing

Save a checkpoint dict containing:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_dice': val_dice,
    'val_iou': val_iou,
    'config': config_dict,
}
```

One `best.pth` per experiment (overwritten when val Dice improves), plus `last.pth` (always latest).

### 5.3 LR Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
)
# Alternatively: CosineAnnealingLR for a fixed budget
```

### 5.4 Reproducibility

```python
# In utils.py
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Seed is read from the YAML config and set at the top of every training script.

### 5.5 Test-Set Evaluation

After each experiment, `evaluate.py` loads `best.pth` and runs inference on the **held-out test set** (never seen during training or hyperparameter tuning):
- Reports the full metrics table (Dice, IoU, Sensitivity, Specificity, Precision, F1, Accuracy)
- Saves a CSV: `results/<experiment>/test_metrics.csv`
- Saves visual prediction overlays: `results/<experiment>/predictions/`

---

## Part 6 — Hardware Verification (Before Any Training)

`scripts/check_environment.py` will run before any training and output a go/no-go report:

```
=== VISTA-XL Environment Check ===

[✓] Python: 3.x.x
[✓] PyTorch: 2.x.x
[✓/✗] CUDA available: True/False
[✓/✗] GPU: <name>, <VRAM> GB
[✓/✗] Dataset path training_data/ : EXISTS, N images found
[✓/✗] Dataset path validation_data/: EXISTS, N images found
[✓/✗] Dataset path test_data/      : EXISTS / NOT FOUND
[✓/✗] Estimated batch memory for UNet (256×256, bs=8): ~X GB  → FITS / OOM RISK
[✓/✗] Estimated batch memory for Swin-UNet (224×224, bs=8): ~X GB → FITS / OOM RISK
[✓/✗] Sample tensor shape check: image (3, 256, 256), mask (1, 256, 256) ✓
```

Memory estimation uses:
```python
# Rough formula: bytes = B * C * H * W * 4 (float32) * num_feature_scales
# This is a lower bound — actual usage includes activations and gradients
```

If VRAM is insufficient, the script suggests safe batch sizes and whether gradient checkpointing is needed.

---

## Part 7 — Experiment Specifications

### Experiment 1: U-Net Baseline

**Purpose:** Establish a reproducible, correctly-evaluated baseline on ISIC 2017 for comparison.

**Model:** `UNet` from the notebook (fixed, not `UNet2`) with the following corrections applied in `src/models/unet.py`:
- No internal normalization
- All bugs from the inspection report addressed
- Output: raw logits (correct for `BCEWithLogitsLoss`)

**What is reused from the notebook:**
- `UNet` class architecture (encoder/decoder structure, `DoubleConv`, skip connections) — reusable with fixes
- `Dataset_Class` concept — rewritten cleanly in `src/dataset.py`
- Metric function logic — rewritten correctly in `src/metrics.py`

**What is new:**
- Proper validation loop with Dice/IoU at every epoch
- Best-model checkpointing
- YAML config, reproducibility seed
- BCEDiceLoss

**Hyperparameters (starting point, tunable):**

| Param | Value |
|---|---|
| Resolution | 256×256 |
| Batch size | 8 (adjust based on hardware check) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-5 |
| Scheduler | ReduceLROnPlateau (patience=10) |
| Epochs | 100 |
| Loss | BCE + Dice (α=0.5) |
| Seed | 42 |

**Expected outputs:** `checkpoints/unet/best.pth`, `results/unet/test_metrics.csv`

---

### Experiment 2: Swin-UNet Baseline

**Purpose:** Establish a transformer-based segmentation baseline before any xLSTM modification.

> **⚠️ Important architectural note:** Swin-UNet is architecturally distinct from TransUNet. TransUNet (in the notebook) uses a ResNet CNN encoder feeding into a 12-block ViT at the bottleneck. Swin-UNet uses a **Swin Transformer as the encoder AND decoder**, with patch merging replacing pooling and patch expanding replacing transposed convolutions. These are different models and their results are not interchangeable.

#### Swin-UNet Architecture Summary

```
Input (B, 3, 224, 224)
  │
  ▼
Patch Embedding (4×4 patches → B, 3136, C)
  │
  ▼
Swin Transformer Encoder (3 stages)
  Stage 1: 2 Swin blocks, resolution/1
  Stage 2: 2 Swin blocks + PatchMerging → resolution/2
  Stage 3: 6 Swin blocks + PatchMerging → resolution/4
  │
  ▼
Bottleneck: 2 Swin blocks
  │
  ▼
Swin Transformer Decoder (3 stages with skip connections)
  Stage D1: PatchExpanding + concat skip + 2 Swin blocks
  Stage D2: PatchExpanding + concat skip + 2 Swin blocks
  Stage D3: PatchExpanding + concat skip + 2 Swin blocks
  │
  ▼
Final PatchExpanding (×4 upsampling)
  │
  ▼
Linear Head → (B, 1, 224, 224) binary mask logits
```

**Key Swin-UNet components** (all require new implementation or a trusted existing codebase):

| Component | Description |
|---|---|
| `SwinTransformerBlock` | Window-based MHSA + Shifted-Window attention + MLP |
| `PatchEmbed` | Splits image into non-overlapping patches |
| `PatchMerging` | Downsampling via concatenation + linear projection |
| `PatchExpanding` | Upsampling via linear projection + rearrange |
| `BasicLayer` | Stack of N `SwinTransformerBlock`s |
| `SkipConnection` | Concatenation of encoder + decoder feature maps |

**Implementation source options (to be decided):**
1. **Implement from scratch** using the original Swin-UNet paper (Cao et al., 2021) — most control, highest effort
2. **Adapt from `timm`** (PyTorch Image Models) SwinTransformer backbone + custom decoder
3. **Use an existing Swin-UNet repo** as a reference, rewriting the key components ourselves

> **⚠️ Recommendation:** Option 2 (timm backbone + custom decoder) balances reproducibility with effort. Option 3 risks incorporating bugs from third-party code. Please confirm approach before implementation.

**What is reused from the notebook:** Nothing directly — Swin-UNet architecture is entirely new. The data pipeline, metrics, loss, and trainer from Experiment 1 are reused.

**Hyperparameters (starting point):**

| Param | Value |
|---|---|
| Resolution | 224×224 |
| Window size | 7×7 (standard) |
| Embed dim | 96 (Swin-T scale) — may need to scale down for VRAM |
| Depths | [2, 2, 6, 2] |
| Num heads | [3, 6, 12, 24] |
| Batch size | 4–8 (hardware-dependent) |
| Pretrained weights | Swin-T ImageNet-1k (optional, confirm with user) |

---

### Experiment 3: VISTA-XL — Swin-UNet + xLSTM Integration

**Purpose:** Investigate whether replacing or augmenting the Swin Transformer's self-attention in the bottleneck/decoder with an xLSTM block improves segmentation accuracy and/or efficiency.

#### What is xLSTM?

xLSTM (Extended LSTM, Beck et al., 2024) extends classical LSTM with:
- **mLSTM (matrix memory):** Replaces scalar cell state with a matrix, enabling exponential gating
- **sLSTM (scalar memory):** conventional LSTM with revised gating (log-gating, normalization)
- **xLSTM blocks:** Stack of mLSTM and sLSTM blocks, designed as a sequence model competitive with Transformers

xLSTM is a **sequence model** designed for 1D token sequences. Applying it to 2D image patches requires flattening spatial tokens — the same approach used to apply ViT/Swin to images.

#### Proposed VISTA-XL Integration Strategy

> **⚠️ This is a tentative architecture — requires your confirmation before implementation.**

**Strategy A — Bottleneck Replacement (Conservative):**
Replace the Swin Transformer bottleneck (2 Swin blocks at lowest resolution) with an xLSTM block:
```
Swin Encoder → [xLSTM Bottleneck] → Swin Decoder (with skip connections)
```
Pros: Minimal change, easiest to ablate, lowest risk.
Cons: Limited novelty if xLSTM only touches the bottleneck.

**Strategy B — Decoder Augmentation (Moderate):**
Keep the Swin encoder intact. Replace the Swin Transformer blocks in the decoder stages with xLSTM blocks:
```
Swin Encoder → Swin Bottleneck → [xLSTM Decoder stages] (with Swin skip connections)
```
Pros: Tests whether xLSTM's sequential inductive bias aids reconstruction.

**Strategy C — Hybrid Attention (Ambitious):**
Within each Swin block, replace or augment window self-attention with an xLSTM layer operating on the flattened window tokens:
```
LayerNorm → [xLSTM(window_tokens)] → (+ residual) → MLP → (+ residual)
```
Pros: Deep integration, potentially stronger novelty claim.
Cons: Highest implementation complexity, harder to debug.

> **⚠️ Q7 (new):** Which strategy do you want to pursue for VISTA-XL? Starting with Strategy A is recommended — it provides a clean ablation baseline and can be extended.

#### xLSTM Implementation

> **⚠️ Q4 applies here.** Two options:

**Option A — Use official `xlstm` PyPI package** (available from `xlstm@github.com/NX-AI/xlstm`):
- Faster integration
- Adds a dependency that may have version conflicts
- Less educational / transparent

**Option B — Implement mLSTM/sLSTM from scratch** based on the paper:
- Full control over architecture
- Can be optimized for 2D spatial tokens
- Significant implementation effort (~500–1000 lines)

**Recommended for a research project:** Option B for the core `mLSTM` block (the novel component) + Option A as a verification reference.

#### What VISTA-XL Does NOT Claim
- It is **not a simple attention module addition**. A genuine xLSTM integration requires replacing the attention mechanism, not adding a module in parallel.
- It is **not proven to be better** until experiments are run. The plan only structures the experiments to test the hypothesis fairly.

---

## Part 8 — Reuse Audit: Notebook vs New Code

| Notebook Component | Reuse Decision | Action Required |
|---|---|---|
| `UNet` class | ✅ Reuse (with bug fixes) | Port to `src/models/unet.py`, fix UNet2 bugs, remove dead code |
| `AttU_Net` | ⚠️ Optional reuse | Could add as a mid-point baseline; low priority |
| `R2U_Net` | ⚠️ Optional reuse | Same as above |
| `TransUNet` | ❌ Do not reuse as Swin-UNet substitute | Architecturally distinct; keep as reference only |
| `UNet2` | ❌ Do not reuse | Too many bugs, non-standard design |
| `MHCA` block | ⚠️ Evaluate separately | Interesting attention module; could be included as an ablation component but is not xLSTM |
| `Dataset_Class` | ✅ Reuse concept, rewrite cleanly | Fix transform application, mask thresholding, dtype casting |
| `get_train_loader` / `get_val_loader` | ✅ Reuse concept | Port to `src/dataset.py` with configurable paths from YAML |
| Metric functions | ✅ Reuse concept, fix formulas | `get_DC` formula is wrong — rewrite all in `src/metrics.py` |
| `Option` / `Config` class | ❌ Replace | Use YAML + dataclass/argparse instead |
| `encode_and_save` / RLE | ❌ Skip | Kaggle submission format, not needed for VISTA-XL |
| `compute_iou` | ❌ Skip | Hardcoded DSB2018 paths; superseded by `src/metrics.py` |
| Training loop | ❌ Replace | No validation, no checkpointing — rewrite entirely in `src/trainer.py` |

---

## Part 9 — Dependency List (Minimal)

```
# Core
torch>=2.0.0
torchvision>=0.15.0

# Data & Augmentation
albumentations>=1.3.0
opencv-python-headless>=4.7.0  # cv2, headless avoids display deps
Pillow>=9.5.0
numpy>=1.24.0

# Config & Utilities
PyYAML>=6.0
tqdm>=4.65.0

# Metrics
scikit-learn>=1.2.0  # For stratified splitting only

# Visualization
matplotlib>=3.7.0

# Experiment tracking (optional but recommended)
tensorboard>=2.13.0
# OR: wandb>=0.15.0  (requires free account)

# xLSTM (if Option A chosen for Q4)
# xlstm  ← confirm version after Q4 decision

# Transformer backbone (if timm approach chosen for Swin-UNet)
timm>=0.9.0
einops>=0.7.0

# Notebook reference environment
pandas>=2.0.0
scikit-image>=0.21.0
```

**Packages explicitly removed vs. notebook:**
- `torch.autograd.Variable` — deprecated since PyTorch 0.4, unused
- `skimage.transform.resize` — replaced by albumentations
- `sklearn.model_selection.train_test_split` — replaced by a clean splitter in `src/dataset.py`

---

## Part 10 — Phased Implementation Timeline

```
Phase 0: Environment & Data Verification   [prerequisite, ~0.5 day]
  - Resolve Q1–Q6 above
  - Run check_environment.py
  - Verify dataset structure and count samples
  - Confirm GPU memory budget

Phase 1: Infrastructure Setup              [~1 day]
  - Create project structure
  - Write requirements.txt
  - Implement src/dataset.py (Dataset class + loaders)
  - Implement src/transforms.py
  - Implement src/metrics.py (correct Dice, IoU, etc.)
  - Implement src/losses.py (BCEDiceLoss)
  - Implement src/utils.py (seeding, env check)
  - Implement src/trainer.py (train loop + val loop + checkpointing)
  - Write train.py and evaluate.py entry points
  - Run 01_data_exploration.ipynb to verify data pipeline

Phase 2: Experiment 1 — U-Net Baseline    [~1–2 days]
  - Port + fix UNet to src/models/unet.py
  - Create configs/unet_baseline.yaml
  - Train full 100 epochs
  - Evaluate on test set, record metrics
  - MILESTONE: Have a validated Dice/IoU number on ISIC 2017

Phase 3: Experiment 2 — Swin-UNet        [~3–5 days]
  - Implement src/models/swin_unet.py
  - Create configs/swin_unet.yaml
  - Verify forward pass with synthetic input
  - Train and evaluate
  - MILESTONE: Swin-UNet Dice/IoU vs U-Net comparison

Phase 4: Experiment 3 — VISTA-XL         [~5–10 days]
  - Implement src/models/xlstm_block.py
  - Integrate into Swin-UNet per chosen strategy
  - Implement src/models/vista_xl.py
  - Ablation experiments: with/without xLSTM at different positions
  - MILESTONE: VISTA-XL Dice/IoU vs both baselines

Phase 5: Analysis & Documentation          [~1–2 days]
  - Complete 02_results_analysis.ipynb
  - Generate prediction visualizations
  - Write README.md
  - Prepare summary metrics table
```

---

## Part 11 — Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU VRAM insufficient for Swin-UNet | Medium | High | Reduce embed_dim to Swin-Tiny; use gradient checkpointing; reduce batch size |
| Dataset path mismatch (local vs Kaggle structure) | Medium | High | `check_environment.py` catches this before training starts |
| xLSTM integration causes training instability | Medium | Medium | Start with Strategy A (bottleneck only); add gradient clipping and careful init |
| Swin-UNet implementation bugs | High | Medium | Verify with unit tests on synthetic input; check forward pass shapes at each stage |
| ISIC 2017 test set not available locally | Low | Medium | Fall back to a programmatic 70/15/15 split |
| xLSTM dependency conflicts | Low | Medium | Use a pinned version; consider implementing from scratch |

---

## Approval Checklist

Before any code is written, please confirm:

- [ ] **Q1:** Dataset location and structure confirmed
- [ ] **Q2:** Split strategy selected (official ISIC splits vs programmatic)
- [ ] **Q3:** GPU name and VRAM provided
- [ ] **Q4:** xLSTM source selected (official library vs from scratch)
- [ ] **Q5:** Target resolution confirmed (128×128 / 224×224 / 256×256)
- [ ] **Q6:** Project scope confirmed (prototype vs publication-quality)
- [ ] **Q7:** VISTA-XL integration strategy selected (A / B / C)
- [ ] **General:** This plan approved to proceed to Phase 0

---

*Plan prepared for: VISTA-XL Research Project*
*Author assistance: Antigravity AI*
*No code has been written or modified as part of this document.*
