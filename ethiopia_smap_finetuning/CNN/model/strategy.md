# CNN Training Strategy

Supervised multi-region drought classification. For each calendar month and each of 27 agro-ecological zones across 5 Horn of Africa countries, predict one of three drought severity classes:

| Class | SPI threshold | Label |
|---|---|---|
| 0 | SPI > −0.5 | Normal |
| 1 | −1.5 < SPI ≤ −0.5 | Moderate drought |
| 2 | SPI ≤ −1.5 | Severe drought |

Targets are derived from CHIRPS v3.0 monthly precipitation via SPI-3 (or SPI-6).  Class distribution in the training set is heavily imbalanced: **89% normal, 8% moderate, 3% severe**.

---

## Data pipeline

### Inputs

Each sample is one `(country, month)` pair:

```
image  (9, 64, 64)      8 SMAP channels + 1 CHIRPS precip, monthly composite
masks  (6, 64, 64)      bool, region pixel membership (padded to R_MAX=6)
labels (6,)             int64, drought class per region (-100 = no label)
```

### Channels

| Index | Name | Source |
|---|---|---|
| 0 | soil_moisture_am | SMAP AM pass, quality-filtered |
| 1 | soil_moisture_pm | SMAP PM pass, quality-filtered |
| 2 | surface_temp_am | SMAP AM |
| 3 | surface_temp_pm | SMAP PM |
| 4 | vegetation_water | SMAP AM |
| 5 | vegetation_opacity | SMAP AM |
| 6 | tb_polarization_diff | tb_v − tb_h (AM) |
| 7 | bulk_density | SMAP AM |
| 8 | precip_mm | CHIRPS v3.0, block-avg resampled to SMAP grid |

### NaN handling

SMAP has ~27% NaN (orbital gaps); CHIRPS has NaN over ocean. Before normalisation every NaN is replaced with the **training-set per-channel mean**. This keeps the pixel grid dense without introducing a hard zero discontinuity.

### Normalisation

Per-channel z-score computed exclusively on the training set, then applied to val and test:

```python
img_norm = (img - channel_mean[:, None, None]) / channel_std[:, None, None]
```

### Temporal split

Strict chronological split — no shuffling across time boundaries — to prevent SPI autocorrelation leakage:

| Split | Period | Images |
|---|---|---|
| Train | 2015-04 → 2022-12 | 465 |
| Val | 2023-01 → 2023-12 | 60 |
| Test | 2024-01 → 2026-03 | 135 |

---

## Model architecture

**CIFAR-style ResNet-18** with multi-region masked pooling.

```
Input (B, 9, 64, 64)
    ↓
Stem: Conv2d(9→64, 3×3, stride=1) + BN + ReLU         → (B, 64,  64, 64)
Layer1: 2× BasicBlock(64→64,   stride=1)               → (B, 64,  64, 64)
Layer2: 2× BasicBlock(64→128,  stride=2)               → (B, 128, 32, 32)
Layer3: 2× BasicBlock(128→256, stride=2)               → (B, 256, 16, 16)
Layer4: 2× BasicBlock(256→512, stride=2)               → (B, 512,  8,  8)
    ↓
Masked Region Pool (per region r):
    downsample mask (64×64) → (8×8) via nearest interpolation
    pooled[r] = sum(feat × mask[r]) / count(mask[r])   → (B, R, 512)
    ↓
Shared FC head (applied independently to each of R region vectors):
    Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→3)
    ↓
Output (B, R, 3) logits
```

**Why CIFAR-style (no initial stride/pool):** with a 64×64 input, the ImageNet stem (7×7 stride-2 + maxpool) would reduce spatial resolution to 8×8 before any residual blocks, producing a 2×2 feature map at the end — too coarse for masked pooling. The CIFAR stem preserves spatial resolution, giving an 8×8 feature map where each cell has a receptive field of ≈60×60 pixels.

**Why masked pooling at the output, not masked input:** zeroing non-region pixels before the backbone discards spatial context (neighbouring moisture gradients, terrain boundaries) that shapes the learned features. The mask is applied only at the aggregation step so convolutions see the full window.

**Trainable parameters:** ~4.2 M (backbone ~4.1 M + head ~66 K)

---

## Training

### Loss

Weighted cross-entropy with inverse-frequency class weights:

```python
weight[c] = N_total / (3 × count[c])
# approximate weights: normal≈0.37, moderate≈4.1, severe≈12.8
```

`ignore_index=-100` skips padded regions and early SPI months where labels are unavailable.

### Optimiser

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clip | max_norm=1.0 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=8) |
| Early stopping | patience=20 epochs on val macro-F1 |
| Dropout | 0.3 (FC head only) |

### Primary metric

**Macro-averaged F1** across the three drought classes. Accuracy is misleading given the 89% normal baseline — a model that always predicts normal achieves 89% accuracy but 0% recall on droughts. Macro-F1 weights all three classes equally.

### Checkpoint

Best model is saved to `CNN/model/checkpoints/best.pt` whenever val macro-F1 improves. The file includes the model weights, channel normalisation stats, and training arguments so inference can be run without reprocessing the data.

---

## Evaluation

Final test-set evaluation uses `sklearn.metrics.classification_report` reporting per-class precision, recall, and F1 for all three drought classes.

---

## Files

```
CNN/model/
    model.py     — ResNet18Backbone, MaskedRegionPool, DroughtCNN
    dataset.py   — DroughtDataset, make_dataloaders
    train.py     — training loop, test evaluation
    checkpoints/ — best.pt, history.json  (created at runtime)
```

## Run

```bash
# First build the country cubes if not yet done:
~/anaconda3/bin/python3 CNN/SMAP_regions/build_country_cubes.py

# Then train:
~/anaconda3/bin/python3 CNN/model/train.py

# SPI-6 target:
~/anaconda3/bin/python3 CNN/model/train.py --label_col drought_class_spi6
```
