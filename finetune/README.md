# `finetune/` pipeline — technical notes

Comparison of four coronary-artery segmentation pipelines on the **ARCADE** dataset (26 classes,
512×512 greyscale X-ray images). The central question: **does a latent representation compressed by
MedVAE allow a segmentation as effective as working directly on the full-resolution image?**

Entry point: `python -m finetune.train --config finetune/configs/condition_{a,b,c,d}.yaml` (always
from the repository root). Each run creates its own folder `experiments/runs/<date>_<name>/` (exact
config, commit, GPU, history, scores); every experiment and its results are in
[`experiments.md`](../experiments.md).

---

## 1. Common base of all conditions

Everything below is shared by conditions A, B, C and D — only the **model** changes from one
condition to the next. The core logic lives in [`trainer/trainer.py`](trainer/trainer.py).

### 1.1 Data and augmentation ([`dataset.py`](dataset.py))

- Images loaded in greyscale (`convert("L")`), normalised to **`[0, 1]`** (`/ 255.0`). The mask is
  rebuilt from the COCO annotations (RLE → `category_id` per pixel).
- **Split** (`data.validation: official`, E08): training on the **1000** images of `seg_train`,
  model selection on the **official `seg_val`** (200 images). The **test set is a separate dataset**
  (ARCADE final phase, 300 images) — never seen during training. The former 80/20 split of
  `seg_train` by seed is still available (`data.validation: split`).
- **Augmentation** (training only, Albumentations, image and mask in sync): `HorizontalFlip(0.5)`,
  `VerticalFlip(0.2)`, `ShiftScaleRotate` (shift 0.05, scale 0.1, rotate ±15°, p=0.7),
  `RandomBrightnessContrast(0.2/0.2, p=0.5)`, light `GaussNoise` (standard deviation 3–7 / 255,
  p=0.3, `data.gauss_noise_std_range`, E07; without parameters, albumentations ≥ 2 adds a noise of
  ~65–84 / 255), `CLAHE(0.3)`. Validation and test only get a `Resize`.
- **Realistic degradations** (option, E15 then E17, off by default): `data.degradation_aug_p: 0.3`
  applies to 30 % of the images one degradation among Poisson noise, JPEG (quality 5–95) and
  Gaussian blur (kernel 3–31). On A (E17, 3 seeds): robustness to degraded images largely restored,
  +0.010 Dice on clean images (see [experiments.md](../experiments.md)).

### 1.2 Loss function ([`losses/seg_loss.py`](losses/seg_loss.py))

`SegLoss` is a **weighted combination of three terms**:

```
L = dice_weight · L_Dice + ce_weight · L_CE + cl_weight · L_clDice
```

| Term | Role | Implementation detail |
|------|------|-----------------------|
| **Dice** | region overlap, robust to class imbalance | `smp.losses.DiceLoss` mode `multiclass`, `from_logits=True`, `smooth=1e-6` |
| **Cross-entropy** | per-pixel classification, stable gradients | standard `nn.CrossEntropyLoss` over the 26 classes |
| **clDice** | **preserves the connectivity / topology** of the vessels (thin arteries) | computed on the foreground classes only (`[:, 1:]`), through a soft morphological skeleton |

**Why clDice?** The standard Dice can be high while an artery is *cut in two*: a few missing pixels
barely penalise the overlap but break the connectivity — a serious error for vessels. clDice
compares the **skeletons** (`soft_skel`, obtained by differentiable morphological erosions and
dilations, `skel_iters=5`) of the predicted and target masks, and measures a topological precision /
sensitivity. It pushes the network towards **continuous** structures.

> **Current setting**: in the four configs, `dice_weight = ce_weight = 0.5` and **`cl_weight = 0.0`**.
> clDice is therefore **implemented but turned off** (set to 0 to save GPU memory — the one-hot
> skeleton over 26 classes is expensive). To turn it on: set `cl_weight` to ~0.2 in the YAML.

### 1.3 Evaluation metrics ([`metrics/seg_metrics.py`](metrics/seg_metrics.py))

The reported **score** is *not* the loss but `torchmetrics` metrics computed on the `argmax`
predictions.

All means are computed over the whole evaluated set (accumulated pixels), then averaged over the
**classes present** (in the prediction or the ground truth).

- **Dice mean** = `MulticlassF1Score(average="macro")` (the Dice is the F1 score in segmentation),
  **background included**. This is the metric used to select the best model.
- **IoU mean** = `MulticlassJaccardIndex(average="macro")`, same averaging. *(Before E01, `MeanIoU`
  received index masks instead of the expected one-hot masks: the originally published IoUs are
  wrong.)*
- **Artery Dice / IoU** (`dice_fg_mean`, `iou_fg_mean`, E09): same averages over the 25 artery
  classes only, **background excluded** (ARCADE standard). The easy background (Dice ~0.99) inflates
  `dice_mean` by about 0.02.
- **Per-class Dice / IoU**: vectors of 26 values in `results.json`.
- A finished run can be re-evaluated with the current metrics:
  `python -m finetune.evaluate experiments/runs/<run>`.

> ⚠️ Important distinction: `dice_loss` (in the loss, differentiable, on logits) ≠ `dice_mean`
> (evaluation metric, on `argmax`, not differentiable). The history curves plot both.

### 1.4 Optimiser, scheduler and training loop

| Element | Value | Note |
|---------|-------|------|
| **Optimiser** | `AdamW` | **filtered**: `filter(lambda p: p.requires_grad, ...)` → the frozen MedVAE parameters are **excluded** from the optimiser |
| **Weight decay** | `1e-4` | L2 regularisation |
| **Scheduler** | `CosineAnnealingLR` | see below |
| **Gradient clipping** | `clip_grad_norm_ = 1.0` | bounds the gradient norm, stabilises training |
| **AMP** | `autocast` + `GradScaler` in **bfloat16** | enabled **only with CUDA** — the code also runs on CPU without error |
| **Early stopping** | patience 15 epochs | monitors the validation `dice_mean` |
| **Max epochs** | 100 | |
| **Checkpoint** | `experiments/runs/<run>/best_model_{experiment_name}.pth` | one folder per run → nothing is overwritten (not versioned) |
| **Seed** | `experiment.seed` | fixes torch, numpy, random and, through `worker_init_fn`, the augmentations of each worker (before the fix made after the audit, the 4 workers replayed the same sequence of augmentations at every epoch). `cudnn.deterministic` is on, but some CUDA operations (backward of the bilinear interpolation) remain non-deterministic: two runs with the same seed can differ slightly |

#### Cosine annealing — how and why

The scheduler is **`CosineAnnealingLR`** (`scheduler.name: "cosine"` in every YAML), stepped **once
per epoch** (`_step_scheduler`).

**How it works.** The learning rate follows half a cosine period, from the initial LR `η_max` (the
`learning_rate` of the config) down to a floor `η_min` (`eta_min: 1e-6`), over `T_max = 100` epochs:

```
η_t = η_min + ½ (η_max − η_min) · (1 + cos(π · t / T_max))
```

- at the start (`t=0`): `η_t = η_max` → large steps, fast exploration;
- in the middle: a smooth decay, slower and slower;
- at the end (`t=T_max`): `η_t = η_min` → tiny steps, fine adjustment.

**Why this choice.**
1. **Smooth decay** rather than a staircase (vs `StepLR`): no sudden drop of the LR that would
   destabilise convergence.
2. **Large LR at the start** to leave bad regions quickly, **small LR at the end** to settle
   precisely in a minimum — what pixel-accurate segmentation boundaries need.
3. No hyperparameter to tune along the way (unlike `ReduceLROnPlateau`, which is *implemented as an
   alternative* in the code but not used by default).

> A `ReduceLROnPlateau` alternative (mode `max` on the validation Dice, `factor 0.5`, `patience 7`)
> is coded in `_build_scheduler` but turned off — set `scheduler.name: "plateau"` to use it.

---

## 2. The conditions

The MedVAE spatial factor matters here: **`medvae_4_1_2d` compresses 4× per spatial dimension**
(not 16×). A `512×512` image → a **`128×128×1`** latent; getting back to 512 takes **2 ×2
upsamplings** (`n_upsample: 2`). *(Naming detail: the "4" in the name is the per-dimension factor;
the paper calls the same model `f=16` because 16 = 4² is the area factor.)*

The `[0,1] → [-1,1]` normalisation required by MedVAE (`x*2−1`) is done **only once**, in
[`encoder/medvae_encoder.py`](encoder/medvae_encoder.py).

| Cond. | Pipeline | Trained model | Scientific question |
|-------|----------|---------------|---------------------|
| **A** | U-Net on the 512×512 image | full U-Net | High-resolution reference |
| **B** | frozen MedVAE (HF) → U-Net head on the latent | head (7.8 M params) | Is general-purpose compression enough? |
| **C** | frozen MedVAE (fine-tuned on ARCADE) → same head | head (7.8 M params) | Does fine-tuning the compressor help? |
| **D** | frozen MedVAE (encode→decode) → U-Net | full U-Net | Does adapting to the compression artefacts compensate? |

### Condition A — U-Net reference ([`models/unet.py`](models/unet.py))

- **SMP U-Net**, ImageNet-pretrained `resnet34` encoder, `in_channels=1`, 26 classes, raw logits as
  output (no softmax — it is in the loss).
- Works directly on the original **512×512** image.
- `batch_size 8`, `lr 1e-4`. All weights are trainable.
- **Role**: reference upper bound. How far can one go without any compression?

### Condition B — frozen pre-trained MedVAE + U-Net head on the latent

- **Encoder**: `MedVAEEncoder` loaded from the HuggingFace weights (`checkpoint_path = null`),
  **fully frozen** (`requires_grad=False`, `eval()`).
- **Head**: `LatentUNetHead` ([`models/seg_head.py`](models/seg_head.py), E04) — a small U-Net at the
  latent resolution (`128 → 64 → 32 → 16 → 128`, skip connections, `base_channels=64`) to capture the
  global context needed to identify the segments, then 2 `interpolate ×2 + ConvBlock` blocks
  (`n_upsample=2`) up to 512 and a 1×1 convolution to 26 classes (7.8 M parameters). The former
  `SegHead` head (1×1 projection then upsampling, no convolution at the latent resolution) is still
  available (`model.architecture: seg_head`).
- `batch_size 4`, **`lr 1e-3`** (E03: with 5e-5, the head stayed stuck on "everything is
  background"). Only the head is trained.
- **Role**: does a *general-purpose* medical compressor (the 2D MedVAE models are trained on chest
  radiographs and mammograms) preserve enough information to segment arteries? B segments the
  latent of MedVAE stage 1 (the reconstruction latent), not the output of the stage-2 projection
  layers.

### Condition C — frozen MedVAE fine-tuned on ARCADE + U-Net head on the latent

- **Same architecture as B**, except for the checkpoint, given at launch:
  ```bash
  python -m finetune.train --config finetune/configs/condition_c.yaml \
    --set encoder.checkpoint_path=experiments/runs/<medvae_finetune run>/best_medvae_finetuned.pth
  # or: MEDVAE_CKPT=<...>/best_medvae_finetuned.pth sbatch finetune/slurm/train_c.sbatch
  ```
  → the encoder loads the MedVAE weights **fine-tuned on ARCADE** (see §3), then is frozen as in B.
  Without a checkpoint, training stops with an error (instead of silently running as B).
- **Role**: isolates the effect of fine-tuning the compressor. Everything else (head,
  hyperparameters) being identical to B, **B vs C** measures exactly the gain brought by adapting
  MedVAE to the coronary domain.

### Condition D — frozen MedVAE autoencoder (encode→decode) + U-Net

- **`MedVAEAutoencoder`**: the image goes through the full cycle
  `[0,1] → [-1,1] → encode → 128×128 latent → decode → [-1,1] → [0,1]`, **frozen**. The output is a
  **reconstructed 512×512 image** (hence degraded by the compression), with the **same interface** as
  the original.
- A **trainable U-Net** (same architecture as A) placed *after* it learns to segment these
  reconstructed images.
- `batch_size 8`, `lr 1e-4`. MedVAE processes the images in groups of 2 (its attention at 512×512
  costs ~1 GB per image; no effect on the result).
- **Role**: the U-Net *sees the compression artefacts from training onwards* and can adapt to them.
  **A vs D** measures the pure cost of compression at equal network capacity.

### Condition A* — evaluation variant ([`eval_astar.py`](eval_astar.py))

Not a training: the **U-Net of condition A** (trained on *original* images) is evaluated on images
**reconstructed by MedVAE**
(`CHECKPOINT_A=experiments/runs/<run A>/best_model_condition_a_unet.pth sbatch finetune/slurm/eval_astar.sbatch`).
It measures the **domain shift**: does a network trained at full resolution absorb the degradation at
test time? To be compared with D, where the network was *trained* on the reconstructed images.

### Controls R and A↓↑ — trivial compression at the same rate (E16)

- **R** ([`configs/condition_r.yaml`](configs/condition_r.yaml), `slurm/train_r.sbatch`): the head of
  B, same architecture and same training, applied to the image averaged over 4×4 blocks (128×128×1,
  as many values as the latent). **B vs R** tells whether the MedVAE latent brings more than a
  trivial reduction.
- **A↓↑** ([`eval_resampled.py`](eval_resampled.py)): the U-Net of A, without retraining, on the
  image reduced to 128×128 then enlarged back to 512×512 (bilinear or bicubic). **A* vs A↓↑** tells
  whether the MedVAE reconstruction preserves more than a resampling.

### Safeguards shared by the frozen MedVAE

- `train()` is **overridden** in `MedVAEEncoder` and `MedVAEAutoencoder`: even when the trainer calls
  `model.train()`, MedVAE stays in `eval()` mode (MedVAE only uses GroupNorm, with no statistics that
  could drift, but `eval()` guarantees the same behaviour as at inference).
- The MedVAE `encode`/`forward` runs under `torch.no_grad()`.

---

## 3. MedVAE fine-tuning — and a reminder of the two stages of the paper

### 3.1 The two training stages of MedVAE (original paper)

MedVAE (Varma et al., MIDL 2025) is a **general-purpose variational autoencoder** pre-trained on
~1.05 M medical images, in **two stages**:

**Stage 1 — base autoencoder (reconstruction quality).** The encoder and decoder are trained end to
end to reconstruct the image, combining **four terms**:
1. **Perceptual loss** (LPIPS) — similarity in the feature space of a pre-trained network, rather
   than pixel to pixel.
2. **Patch adversarial loss** (PatchGAN discriminator) — restores the high frequencies / sharpness
   that an L1 reconstruction alone blurs.
3. **BiomedCLIP embedding consistency** — an **L2 penalty between the BiomedCLIP embeddings** of the
   input and reconstructed images, to keep the clinically relevant features.
4. **KL regularisation** — divergence towards a standard Gaussian, with a **very small weight
   (1e-6)** so as not to degrade the reconstruction (this is what makes it a *VAE* and keeps the
   latent regular and usable).

**Stage 2 — preserving clinical features (latent refinement).** For the 2D models, the **encoder
and decoder are frozen**, and **small trainable projection layers** are added on the latent. They are
optimised with a **BiomedCLIP embedding-consistency loss** (L2 between the embedding of the input
image and that of the projected latent representation). Goal: make sure that fine, subtle but
clinically decisive details survive the compression — where a general-purpose VAE would smooth them
out.

> **Why two stages?** Stage 1 gives a compressor that *reconstructs well visually*; stage 2 forces
> the latent to stay *semantically faithful to the clinical content*. A sharp reconstruction ≠ a
> latent informative for a medical task: the two stages cover these two distinct goals.

**Naming of the released models**: `f` = **area** downsampling factor, `C` = latent channels. The 4
2D models are `f16/C1`, `f16/C3`, `f64/C1`, `f64/C4`. The `medvae_4_1_2d` used here = `f16, C1` (the
"4" = factor **per dimension**, 16 = 4²).

### 3.2 Fine-tuning on ARCADE ([`finetune_medvae.py`](finetune_medvae.py))

Before condition C, **the whole MedVAE is re-adapted** to the coronary domain — a light and
deliberately simple fine-tuning (≠ the elaborate stages of the paper):

- **Dataset**: ARCADE images *only* (no mask), resized to 512, normalised `[0,255] → [-1,1]`
  (`/127.5 − 1`). 90 % / 10 % split of `seg_train`.
- **The whole MedVAE is unfrozen** (`requires_grad=True`, `mvae.train()`) — the only moment it is
  trained.
- **Loss = pure `L1`** between the original image and `decode(encode(image))`. No perceptual loss, no
  GAN, no KL: the goal is only to align the reconstruction with the distribution of coronary X-rays.
- **Optimiser**: `AdamW`, **`lr = 4.5e-6`** (very small — fine-tuning, not training *from
  scratch*), `weight_decay = 0`.
- **Scheduler**: `CosineAnnealingLR`, `T_max = 50` (= number of epochs), `eta_min = 1e-7` — the same
  cosine logic as §1.4, adapted to the length of the fine-tuning.
- Batch of 4 processed as 4 micro-batches of 1 with gradient accumulation (`micro_batch_size: 1`) to
  fit on an RTX 3090 (24 GB); identical to the full batch (mean L1, per-image normalisations).
- `grad_clip = 1.0`, **early stopping** with patience 10 on the **validation L1**.
- Saves `best_medvae_finetuned.pth` in the run folder → passed to condition C
  (`--set encoder.checkpoint_path=...` or `MEDVAE_CKPT`).

In use (condition C), `MedVAEEncoder._load_finetuned` reloads these weights (`state_dict` /
Lightning / `Phase1Trainer` formats): the MedVAE keys start with `model.` and not `encoder.`, so the
**whole model** is reloaded (0 missing keys, checked), then everything is frozen again.

---

## 4. Hyperparameter summary

| | Cond. A | Cond. B | Cond. C | Cond. D | MedVAE FT |
|---|---|---|---|---|---|
| Trained model | U-Net r34 | LatentUNetHead | LatentUNetHead | U-Net r34 | full MedVAE |
| MedVAE | — | frozen (HF) | frozen (FT) | frozen (enc→dec) | **trained** |
| Network input | 512 image | 128 latent | 128 latent | 512 reconstructed image | 512 image |
| `batch_size` | 8 | 4 | 4 | 8 | 4 |
| `learning_rate` | 1e-4 | 1e-3 | 1e-3 | 1e-4 | 4.5e-6 |
| `weight_decay` | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 0 |
| Scheduler | cosine T=100 | cosine T=100 | cosine T=100 | cosine T=100 | cosine T=50 |
| Loss | Dice+CE | Dice+CE | Dice+CE | Dice+CE | L1 |
| Validation | official seg_val | seg_val | seg_val | seg_val | 10 % of seg_train |
| Early stop (patience) | 15 (val Dice) | 15 | 15 | 15 | 10 (val L1) |
| AMP bf16 | ✓ (GPU) | ✓ | ✓ | ✓ | — |

---

## Sources

- [MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders (arXiv:2502.14753)](https://arxiv.org/abs/2502.14753)
- [HTML version of the paper](https://arxiv.org/html/2502.14753v1)
- [Official repository StanfordMIMI/MedVAE](https://github.com/StanfordMIMI/MedVAE)
