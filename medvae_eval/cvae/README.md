> **Historical document (first version of the study).** Notebooks 01 to 03 (construction of the
> quality scores) remain valid; the FiLM study of notebooks 04 to 07 was redone (E13, E18:
> `medvae_eval/film/`), and its conclusions are superseded: see
> [experiments.md](../../experiments.md). The folder used to be called `theophile_metrics/`.

# Summary — inductive bias and image quality pipeline (ARCADE)

> This folder contains **7 notebooks** forming a complete pipeline: extraction of no-reference
> quality metrics on the ARCADE coronary images, generation of an inductive bias (3 approaches),
> modification of the MedVAE architecture, robustness evaluation, ablation, and comparison with a
> PSNR masked by the vessel anatomy.

---

## Global configuration — `pipeline_config.py`

A **single hyperparameter** `APPROACH` controls how the quality score `c` is computed throughout the
pipeline. Change it and rerun NB3 → NB4 → NB5 to compare the approaches.

| Approach | Column | Description |
|----------|--------|-------------|
| **A** | `score_weighted` | Linear combination of 6 min-max normalised IQA metrics, weighted by the **absolute PC1 loadings** (`\|loading\| / Σ\|loading\|`, fitted on TRAIN) — objective weights, no arbitrary manual choice |
| **B** | `score_pca` | PCA (normalised PC1), sign corrected through the correlation with Tenengrad |
| **C** | `score_dl` | **Learned score**: MLP (512→128→32→1) trained on frozen ResNet-18 features of synthetically degraded TRAIN images (noise / blur / JPEG), target = `1 - known severity` |

The outputs of NB4, NB5 and NB6 are organised by approach (`approach_A/`, `approach_B/`,
`approach_C/`) for comparison. NB7 compares the 3 approaches at once.

---

## Pipeline overview

```
pipeline_config.py → APPROACH = "A" | "B" | "C"
    ↓
NB1 → Exploration & MSCN
NB2 → Per-patch quality maps (8 metrics × G×G) [+ ResNet-18 features for approach C]
NB3 → Single score c ∈ [0,1] → labels_quality.csv (score_weighted + score_pca + score_dl)
NB4 → QualityAwareAutoencoderKL (FiLM) → approach_{X}/checkpoints/quality_aware_vae.pt
NB5 → Robustness evaluation (baseline vs cVAE) → approach_{X}/{figures,tables}
NB6 → Constant-c ablation (to rerun for A, B and C) → approach_{X}/{figures,tables,checkpoints}
NB7 → Standard PSNR vs masked PSNR (3 approaches + ablation)
quality_metrics.ipynb → Simplified standalone pipeline (11 NR-IQA metrics + composite score)
```

**Rerun order after a change of methodology (A and/or C):** to start again from a fully consistent
state, run `NB3(A)→NB4(A)→NB5(A)`, `NB3(C)→NB4(C)→NB5(C)`, then `NB3(B)→NB4(B)→NB5(B)→NB6(B)`, and
finish with a last run of **NB3 with `APPROACH="C"`** to regenerate `labels_quality.csv` with the 3
score columns at once (NB3 only exports `score_dl` when `APPROACH="C"`), before running NB7.

---

## Notebook 1 — Exploration and preprocessing (`01_Data_Exploration_and_Preprocessing.ipynb`)

**Goal:** understand the quality variability of the ARCADE dataset and prepare the MSCN
normalisation.

### Steps

1. **Sample loading** — 20 random images from `seg_train/images/`, converted to greyscale float32
   [0, 1].
2. **Visual overview** — mosaic of the sample to see the quality variability by eye.
3. **Three basic metrics** — Laplacian variance (sharpness), RMS contrast, Shannon entropy. Sorted by
   increasing sharpness to find the blurriest images.
4. **Good vs bad** — side-by-side figure of the sharpest and the blurriest image (key figure of the
   report).
5. **Metric distributions** — histograms of the 3 metrics over the sample.
6. **MSCN normalisation** (mean subtracted contrast normalised) —
   `Î(i,j) = (I(i,j) - μ(i,j)) / (σ(i,j) + C)` — removes the biases of non-uniform illumination and
   the global contrast biases between patients.
7. **MSCN check** — visual before/after comparison, distribution histograms, and a check that the
   metrics still separate the images after MSCN.

### Outputs (in `outputs_cvae/01_Data_Exploration/figures/`)

| File | Content |
|------|---------|
| `01_panorama_sample.png` | Mosaic of the sample |
| `01_good_vs_bad.png` | Sharp vs blurred image |
| `01_metrics_distribution.png` | Histograms of the 3 metrics |
| `01_mscn_comparison.png` | Effect of the MSCN normalisation |
| `01_histograms_mscn.png` | Pixel distributions before/after MSCN |
| `01_metrics_raw_vs_mscn.png` | Raw vs post-MSCN metrics |

---

## Notebook 2 — Per-patch IQA metric extraction (`02_IQA_Metrics_Extraction.ipynb`)

**Goal:** compute no-reference quality metrics on **the whole ARCADE dataset**, assessing each image
**locally** through a 4×4 grid of patches.

### Metrics (8 per patch)

| Metric | Type | Rationale |
|---|---|---|
| Laplacian variance | Sharpness | Standard, robust on radiographs |
| RMS contrast | Local contrast | Sensitive to motion artefacts |
| Shannon entropy | Disorder / artefacts | Validated on DSA (PatchDSA, 2024) |
| Tenengrad | Global gradient (Sobel L2) | Robust to the thin edges of the arteries |
| Tenengrad H / V / D1 / D2 | Directional gradient (4 kernels) | Detects cardiac motion blur (dominant direction) |

### Steps

1. **Full dataset loading** — all images of `seg_train` + `seg_val`.
2. **Patch split** — 4×4 grid, each image → 16 patches of size `(H/4, W/4)`.
3. **MSCN normalisation** — applied before computing the metrics.
4. **2D quality map** — a `(4, 4, 8)` array per image, holding the 8 metrics of each patch.
5. **Directional analysis** — visualisation of Tenengrad H/V/D1/D2 to detect motion blur (the
   heartbeat mainly degrades the vertical direction).
6. **ResNet-18 features** — a 512-D vector per image from a frozen pre-trained ResNet-18
   (`AdaptiveAvgPool2d` layer), used as input of the learned score (method C, NB3) — only extracted
   when `APPROACH="C"`.
7. **Full dataset processing** — parallelised (4 threads). "Black" patches (radiograph borders, mean
   intensity < 2 %) are excluded. Valid patches are aggregated into mean/std/max/min per metric.
8. **Statistical analysis** — distributions of the 8 metrics over the whole dataset, correlation
   matrix to find redundancies.
9. **Export** — CSV + `.npy` quality maps.

### Outputs (in `outputs_cvae/02_IQA_Metrics_Extraction/`)

| File | Content |
|------|---------|
| `metrics/patch_metrics_full.csv` | 8 metrics × (mean/std/max/min) for each image |
| `quality_maps/*.npy` | `(4, 4, 8)` quality map per image |
| `resnet_features/resnet18_features.npy` | (N, 512) matrix of ResNet-18 features |
| `resnet_features/resnet18_ids.csv` | image_id ↔ row index in the matrix |
| `figures/02_qmap_demo.png` | Quality heatmaps per metric (demo) |
| `figures/02_directional_tenengrad.png` | Directional analysis H/V/D1/D2 |
| `figures/02_full_dataset_distributions.png` | Metric histograms (whole dataset) |
| `figures/02_correlation_matrix.png` | Correlation matrix between metrics |
| `figures/02_extreme_images.png` | Extreme images (worst vs best Tenengrad) |

---

## Notebook 3 — Inductive bias generation (`03_Inductive_Bias_Generation.ipynb`)

**Goal:** merge the 8 IQA metrics of notebook 2 into a **single score c ∈ [0, 1]** usable by the
cVAE (1 = good quality, 0 = degraded image).

### Feature engineering

Two spatial indicators derived from the data of notebook 2:

| Feature | Definition | Interpretation |
|---|---|---|
| `spatial_homogeneity` | `1 - σ_tenengrad / (μ_tenengrad + ε)` | 1 = uniform quality, 0 = local blurred regions |
| `directional_balance` | `min(H, V, D1, D2) / (max(H, V, D1, D2) + ε)` | 1 = isotropic, 0 = directional blur (motion) |

### Train/val separation (no leakage)

A `split` column is added when loading the metrics, derived from the image path (`seg_train` →
`"train"`, `seg_val` → `"val"`). **Every statistical operation (clipping, normalisation, PCA, MLP
training, thresholds) is fitted on the TRAIN images only**, then applied (transform) to the whole
dataset.

### Three scoring methods compared

| Method | Principle | Advantage |
|---|---|---|
| **B — PCA (PC1)** *(computed first)* | First principal component normalised to [0, 1] (**PCA fitted on TRAIN**), sign corrected through the correlation with Tenengrad (on TRAIN) | Objective, data-driven, no arbitrary choice of weights |
| **A — Weighted score** | Linear combination of 6 min-max normalised features (**fitted on TRAIN**), with weights = **absolute PC1 loadings of method B**, normalised to sum to 1 | Physically interpretable, objective weights (no more manual values) |
| **C — Learned score** | MLP (512→128→32→1, sigmoid output) trained with MSE on the frozen ResNet-18 features of **synthetically degraded TRAIN images** (Gaussian noise, motion blur, JPEG), target = `1 - known severity`. Then applied to the real ResNet-18 features (NB2) and min-max renormalised (TRAIN bounds) | Uses a learned (non-linear) representation, not only a linear combination of classical metrics |

> **Caveat (method C):** ARCADE provides no real quality annotation. The MLP's training target is a
> **synthetic proxy** (known degradation severity), not a human ground truth. The MLP's internal
> train/val split is made by image (never by sample), always within the TRAIN images.

### Final score selection

The final score (`quality_score`) is set by the `APPROACH` hyperparameter in `pipeline_config.py`:
- `APPROACH = "A"` → `score_weighted`
- `APPROACH = "B"` → `score_pca`
- `APPROACH = "C"` → `score_dl`

`score_weighted` and `score_pca` are **always** computed and exported, whatever the active approach.
`score_dl` is only computed (and exported) when `APPROACH="C"` (the ResNet-18 features of NB2 must be
available).

### Categorical labels

The continuous score is discretised into 3 classes by tertiles **computed on TRAIN only**:

| Label | Tertile | Meaning |
|---|---|---|
| `bad` | [0, 33%[ | Severe artefacts |
| `medium` | [33%, 67%[ | Acceptable quality |
| `good` | [67%, 100%] | Clinically usable |

### Validation

- Gallery sorted by score (visual consistency check)
- Correlation of the final score with each raw metric
- Gallery of 3 examples per class (bad / medium / good)
- Method C: MLP training curve, scatter of prediction vs known synthetic severity, `score_dl` vs
  `score_weighted` / `score_pca`

### Outputs (in `outputs_cvae/03_Inductive_Bias_Generation/`)

| File | Content |
|------|---------|
| `labels_quality.csv` | **Shared** file, overwritten at every NB3 run: image_id, path, **split**, quality_score, quality_label, score_weighted, score_pca [, score_dl]. `quality_score` / `quality_label` reflect the **last** approach run (the `score_*` columns are kept between runs). |
| `common/03_engineered_features.png` | Distribution of the spatial_homogeneity and directional_balance features (independent of the approach) |
| `common/03_score_comparison.png` | Correlation and difference between methods A and B (independent of the approach: always compares score_weighted vs score_pca) |
| `approach_{A,B,C}/labels_quality.csv` | **Frozen copy** per approach: same columns as above, but `quality_score` / `quality_label` correspond to THIS approach, even after others have run since. |
| `approach_{A,B,C}/03_label_distribution.png` | Distribution of the final score (of this approach) with thresholds + pie chart of the labels |
| `approach_{A,B,C}/03_gallery_sorted_by_score.png` | Gallery sorted by quality_score (of this approach) |
| `approach_{A,B,C}/03_examples_per_class.png` | 3 examples per class (bad / medium / good), labels of this approach |
| `approach_A_weighted_pca/03_score_weighted.png` | Distribution of the weighted score + PCA weight of each feature |
| `approach_B_pca1/03_pca_biplot.png` | PC1 loadings + PC1 vs PC2 biplot |
| `approach_C_learned/quality_mlp_resnet.pt` | Trained MLP (weights + scaler + metadata) |
| `approach_C_learned/03_dl_training_curve.png` | Training curve + predicted vs known severity on validation |
| `approach_C_learned/03_dl_score_comparison.png` | `score_dl` vs `score_weighted` / `score_pca` |

> ⚠️ `03_label_distribution.png`, `03_gallery_sorted_by_score.png` and `03_examples_per_class.png`
> depend on `quality_score` / `quality_label`, hence on the active approach — they were moved from
> `common/` to `approach_{A,B,C}/` (fixed on 2026-06-17) so that runs of different approaches no
> longer overwrite them.

---

## Notebook 4 — Quality-aware cVAE architecture (`04_MedVAE_Architecture_Mod.ipynb`)

**Goal:** plug the quality score c ∈ [0,1] into the Stanford MedVAE architecture to build a
**quality-aware conditional variational autoencoder (cVAE)** through FiLM conditioning.

### FiLM conditioning (Perez et al., AAAI 2018)

A shared MLP maps the scalar c to a 128-dimensional embedding. Then 19 linear heads (one per
`ResnetBlock` of the encoder + decoder) produce affine parameters (γᵢ, βᵢ). After each
`ResnetBlock`:

```
h ← (1 + γᵢ) · h + βᵢ
```

| Aspect | Channel concatenation | FiLM (chosen) |
|---|---|---|
| Injection point | Input only | Every layer (19 blocks) |
| Model change | Patch `conv_in` (+1 channel) | None (forward hooks) |
| Modulation type | Constant spatial channel | Learned scale + shift per layer |
| Initialisation | Zero | Zero (identity at the start → stability) |
| Added parameters | ~0.01 % | ~2.7 % |

### Steps

1. **Dataset** — `QualityAwareArcadeDataset` returns `(image, quality_score)` instead of
   `(image, mask)`. Images resized to 64×64, rescaled to [-1, 1]. **No leakage: the training
   DataLoader only uses the TRAIN images** (`split` column of the CSV), with a separate validation
   DataLoader.
2. **Architecture** — `QualityAwareAutoencoderKL` wrapping `AutoencoderKL2D` with a
   `FiLMConditioner`. Zero init on every head → no effect at the start.
3. **Checks** — forward pass, gradient flow (∂loss/∂c ≠ 0), parameter count.
4. **Architecture diagram** — annotated figure of the pipeline x → Encoder → z → Decoder → x̂ with
   the FiLM injection.
5. **Pre-training visualisation** — reconstructions at different c (identical, because of the zero
   init).
6. **Phase 1 — FiLM warm-up** (2,500 steps, ~10 epochs) — only the FiLM parameters are trained (base
   VAE frozen). ELBO loss = MSE + KL_weight × KL. Validation every 250 steps with early stopping
   (patience=10).
7. **Post-warm-up visualisation** — reconstruction of a bad and a good image at c=0, 0.5, 1. The
   score starts to modulate the output.
8. **Phase 2 — global fine-tuning** (5,000 steps, ~20 epochs) — all weights are unfrozen (base VAE +
   FiLM). LR lowered to 1e-6 with a cosine scheduler → 1e-7. Same validation and early stopping.
9. **Checkpoint** — `quality_aware_vae.pt` with the best model (selected on the validation loss),
   ddconfig, embed_dim, and the histories of both phases.

### Outputs (in `outputs_cvae/04_MedVAE_Architecture_Mod/approach_{APPROACH}/`)

| File | Content |
|------|---------|
| `checkpoints/quality_aware_vae.pt` | Checkpoint of the cVAE + ablation baseline (model + config + history) |
| `figures/04_architecture_cvae.png` | Diagram of the FiLM architecture |
| `figures/04_conditioning_before_training.png` | Reconstructions at various c (zero init) |
| `figures/04_warmup_curves.png` | Phase 1 loss curves (train + val) |
| `figures/04_conditioning_after_training.png` | Effect of the conditioning after the warm-up |
| `figures/04_finetune_curves.png` | Phase 2 loss curves (train + val) |
| `figures/04_baseline_curves.png` | Loss curves of the ablation baseline (no FiLM) |

---

## Notebook 5 — Robustness evaluation (`05_Evaluation_of_Robustness.ipynb`)

**Goal:** show that the inductive bias improves the reconstruction — the **quality-aware cVAE**
should reconstruct degraded images better than the baseline MedVAE.

### Experimental protocol

| Experiment | Description |
|---|---|
| **A — Real images** | Baseline vs cVAE on the `bad/medium/good` ARCADE images (20 images per class) |
| **B — Synthetic degradations** | Artificially degrade the best images and compare the robustness |

### Reconstruction metrics

| Metric | Description | Direction |
|---|---|---|
| MSE | Pixel-wise mean squared error | ↓ better |
| SSIM | Structural similarity index | ↑ better |
| PSNR | Peak signal-to-noise ratio | ↑ better |
| HaarPSI | Haar perceptual similarity index (Reisenhofer et al., 2018) | ↑ better |

> **Note:** HaarPSI is preferred because it is more reliable than SSIM/PSNR on medical images
> (Breger et al., 2024). Notebook 7 completes this analysis with a **PSNR masked by the vessel mask**
> (see below).

### Grid of synthetic degradations (7 types)

Clean, Gaussian noise (σ=0.05, 0.10, 0.20), horizontal motion blur (k=5), vertical motion blur (k=5,
k=11).

### Steps

1. **Loading both models** — cVAE (from the NB4 checkpoint) + baseline (same base weights, no FiLM).
2. **Test sets** — stratified selection of 20 images per class (bad/medium/good).
3. **Inline quality score** — `inline_quality_score()` recomputes a score c directly from a tensor
   (for synthetic images without a label).
4. **Evaluation on real images** — reconstruction with the true `quality_score` for the cVAE.
5. **Evaluation on synthetic degradations** — metrics computed against the **original clean image**
   (ground truth).
6. **Statistical comparison** — Wilcoxon test (non-parametric) for each metric on the `bad` images.
7. **Visualisations** — box plots per class, bar charts per degradation type, qualitative galleries
   (Original | Baseline | cVAE | Diff).

### Reading the results

- The cVAE is trained with a FiLM warm-up (2,500 steps) + global fine-tuning (5,000 steps) = **7,500
  steps** in total, with validation and early stopping.
- This notebook is run separately for each of the 3 approaches (`approach_A/`, `approach_B/`,
  `approach_C/`).

### Outputs (in `outputs_cvae/05_Evaluation_of_Robustness/approach_{APPROACH}/`)

| File | Content |
|------|---------|
| `tables/05_evaluation_results.csv` | Per-image metrics (baseline + cVAE) on real images |
| `tables/05_synthetic_results.csv` | Metrics per image × degradation on synthetic images |
| `figures/05_degradation_grid.png` | Grid of the 7 degradation types |
| `figures/05_boxplots.png` | Box plots of MSE/SSIM/HaarPSI per quality class |
| `figures/05_synthetic_metrics.png` | Bar charts baseline vs cVAE per degradation |
| `figures/05_gallery_bad.png` | Qualitative gallery on bad images |
| `figures/05_gallery_synthetic.png` | Gallery on synthetically degraded images |

---

## Notebook 6 — Constant-c ablation (`06_Ablation_c_Constant.ipynb`)

**Goal:** check that the signal `c` carries useful information for the cVAE, by comparing
`cvae_{APPROACH}` (trained with the true score of the active approach) with a second cVAE trained
with a constant `c=0.5` for every image — same architecture, same data, same compute budget. If the
cVAE with the real c is significantly better, the signal `c` is used; otherwise, FiLM conditioning
brings nothing beyond extra model capacity. **Generalised to the 3 approaches**: the notebook reads
`APPROACH` from `pipeline_config.py` like NB4/NB5, so it must be rerun once per approach (A, B, C) to
get the full ablation of the 3 — it is no longer limited to B.

### Outputs (in `outputs_cvae/06_Ablation_c_Constant/approach_{APPROACH}/`)

| File | Content |
|------|---------|
| `checkpoints/quality_aware_vae_c_constant.pt` | Checkpoint of the cVAE trained with a constant c=0.5 |
| `tables/06_ablation_results.csv` | Per-image metrics, real c vs constant c |
| `tables/06_synthetic_results.csv` | Metrics on synthetic degradations |
| `figures/06_training_curves.png` | Training curves of the cVAE (c=0.5) |
| `figures/06_boxplots.png` | Metric distributions, real c vs constant c |
| `figures/06_delta_per_image.png` | Per-image Δmetric as a function of the quality score |
| `figures/06_gallery.png` | Qualitative gallery bad/good |
| `figures/06_synthetic_barplot.png` | Comparison on synthetic degradations |

---

## Notebook 7 — Masked PSNR comparison (`07_Masked_PSNR_Comparison.ipynb`)

**Goal:** take up the evaluations of notebooks 5/6 (baseline vs cVAE, 3 approaches + ablation) and
complete the **standard PSNR** with a **masked PSNR** (`medvae_eval/scripts/masked_psnr.py`), which
computes the MSE only on the pixels of the annotated vessel mask (ARCADE COCO polygons), instead of
the whole image.

### Important methodological point

Each model is conditioned on **the native score of its approach** (`score_weighted` for the `_A`
models, `score_pca` for `_B`, `score_dl` for `_C`), read directly from `labels_quality.csv` (which
holds the 3 columns at once). The bad/medium/good classes are **recomputed independently for each
approach** (tertiles on TRAIN), instead of reusing the generic `quality_label` column of the CSV —
which only reflects the last approach NB3 was run with. This avoids any cross-contamination between
approaches during the evaluation.

### Outputs (in `outputs_cvae/07_Masked_PSNR_Comparison/`)

| File | Content |
|------|---------|
| `tables/07_full_results.csv` | Full results (3 approaches + ablation) |
| `tables/07_psnr_comparison_table.csv` | Standard vs masked PSNR, per class and model |
| `tables/07_verdict_flip_table.csv` | Does the winner change with the metric? |
| `tables/07_synthetic_results.csv` | Results on synthetic degradations |
| `figures/07_mask_explanation.png` | Visualisation of the vessel mask (signal vs noise) |
| `figures/07_psnr_vs_masked_bars.png` | Standard vs masked bars per approach |
| `figures/07_psnr_correlation_scatter.png` | Per-image correlation between the two metrics |
| `figures/07_boxplots_*.png` | Distributions per comparison |
| `figures/07_gallery_bad_approach_{A,B,C}.png` | Qualitative galleries per approach, mask overlay |
| `figures/07_gallery_good_ablation.png` | Ablation gallery, real c vs constant c |
| `figures/07_gallery_synthetic_masked.png` | Gallery on synthetic degradations |
| `figures/07_synthetic_psnr_vs_masked.png` | Standard vs masked PSNR by degradation severity |

---

## quality_metrics.ipynb — simplified standalone pipeline

**Goal:** a light, standalone NR-IQA quality pipeline, independent of the NB1→NB7 chain. It directly
produces a composite score per image.

### Metrics (11 no-reference metrics)

| Category | Metric | Direction |
|----------|--------|-----------|
| Sharpness (classical) | Laplacian variance | ↑ = sharper |
| Sharpness (classical) | Tenengrad (Sobel) | ↑ = sharper |
| Sharpness (classical) | Brenner | ↑ = sharper |
| Contrast | RMS contrast | ↑ = more contrast |
| Contrast | Michelson | ↑ = more contrast |
| Exposure | Mean brightness | middle = well exposed |
| Noise (classical) | Immerkaer σ | ↓ = less noise |
| Noise (classical) | Wavelet σ (Haar) | ↓ = less noise |
| Classical NR-IQA | BRISQUE (piq) | ↓ = better quality |
| Classical NR-IQA | NIQE (piq) | ↓ = better quality |
| Deep NR-IQA | CLIP-IQA (piq) | ↑ = better quality |

### Composite score (`quality_score`)

Unweighted mean of 7 min-max normalised metrics:

| Metric | Normalisation |
|--------|---------------|
| `laplacian_var` | direct (↑ = better) |
| `tenengrad` | direct (↑ = better) |
| `rms_contrast` | direct (↑ = better) |
| `clip_iqa` | direct (↑ = better) |
| `immerkaer_noise` | inverted: `1 - minmax(x)` |
| `brisque` | inverted: `1 - minmax(x)` |
| `niqe` | inverted: `1 - minmax(x)` |

**Metrics left out of the composite:**

- **`brenner`** — redundant with `laplacian_var` and `tenengrad` (would overweight sharpness).
- **`michelson`** — sensitive to outliers (a single extreme pixel skews the result).
- **`mean_brightness`** — not a quality metric (no clear good/bad direction).
- **`wavelet_noise`** — redundant with `immerkaer_noise` (would overweight noise).

### Visualisations

- Distribution histograms of each metric
- Correlation heatmap between metrics
- Visual comparison of the 5 lowest vs highest quality images (Laplacian variance)
- Histogram of the composite score

### Outputs

| File | Content |
|------|---------|
| `outputs_cvae/quality_metrics_standalone/quality_metrics_scores.csv` | `image, quality_score` for each image |
| `outputs_cvae/quality_metrics_standalone/quality_distributions.png` | Histograms of the 11 metrics |
| `outputs_cvae/quality_metrics_standalone/quality_correlation.png` | Correlation matrix |
| `outputs_cvae/quality_metrics_standalone/quality_comparison.png` | 5 low quality vs 5 high quality |
| `outputs_cvae/quality_metrics_standalone/quality_composite.png` | Distribution of the composite score |

---

## Utility file — `medvae_standalone.py`

**Goal:** a local, standalone re-implementation of the Stanford (MIMI) MedVAE architecture. It
removes the dependency on the `jepa-adaptation/` folder and makes the notebooks portable.

### Components

| Class / function | Role |
|---|---|
| `DiagonalGaussianDistribution` | VAE posterior (μ, log-σ²). Methods: `sample()` (reparameterisation), `kl()` (KL divergence vs N(0,I)), `mode()` (deterministic). Log-variance clamped to [-30, 20]. |
| `nonlinearity(x)` | Swish/SiLU activation |
| `Normalize(in_channels)` | GroupNorm (32 groups) |
| `Upsample` / `Downsample` | 2× spatial resizing (nearest + optional conv) |
| `ResnetBlock` | Residual block with GroupNorm + Swish + Conv |
| `AttnBlock` | Multi-head self-attention |
| `Encoder` | Downward path: Conv → ResBlocks → latent |
| `Decoder` | Upward path: latent → ResBlocks → Conv → reconstruction |
| `AutoencoderKL` | Full VAE combining Encoder + Decoder + quant_conv |

### Use in the notebooks

```python
from medvae_standalone import AutoencoderKL, DiagonalGaussianDistribution
base_ae = AutoencoderKL(ddconfig=DDCONFIG, embed_dim=EMBED_DIM, ckpt_path=PHASE1_CKPT)
```

This file is imported by **NB4**, **NB5**, **NB6** and **NB7** to load the pre-trained Stanford
weights (`vae_4x_4c_2D.ckpt`) and build the quality-aware cVAE.

---

## Utility file — `medvae_eval/scripts/masked_psnr.py`

**Goal:** defines `MaskedPSNR`, a PyTorch module that computes the PSNR only on the pixels of the
vessel mask (ARCADE COCO segmentation polygons), instead of the whole image. Used by **NB7**.
