# Experiment log

Every change tested on the project, **whether it improved the results or not**. Each change
is compared with the state of the project **just before** it, measured under the same
conditions (same GPU, same dataset, same seed).

## Summary

| # | Part | What was tested | Measured effect | Decision |
|---|---|---|---|---|
| E01 | D | IoU computation (index masks passed to `MeanIoU`) | published IoUs wrong (0.50 → ~0.33) | fixed |
| E02 | D | Reproduction on RTX 3090 | Dice within ±0.006 of the original; B and C detect no artery | reference |
| E03 | D | Head learning rate of B/C: 5e-5 → 1e-3 | B 0.039 → 0.097, C 0.039 → 0.091 | kept |
| E04 | D | U-Net head at the latent resolution (B/C) | B → 0.409, C → 0.398: the latent is usable | kept |
| E05 | D | Adoption of E04 (U-Net head by default for B and C) | — | kept |
| E06 | D | 3 seeds per condition | spread ±0.01; D > A for all 3 seeds | protocol |
| E07 | D | Augmentation noise brought back to the intended strength | A +0.010 (small) | kept |
| E08 | D | Validation on the official seg_val (1000 training images) | A +0.030 | kept |
| E09 | D | Dice / IoU on the arteries only (background excluded) | ~0.024 below the Dice with background, same ranking | added |
| E10 | D | New reference state (E07 + E08) on A, B, C, D, A* | A 0.452, A* 0.448, B 0.417, C 0.427, D 0.455; D ≈ A now | reference, replaced by E17 |
| E11 | A | Robustness measured against the clean image | reduces Poisson noise (+1.2 to +6.6 dB, a 5×5 filter does +6.3 to +10.3), restores neither JPEG nor blur | redone |
| E12 | A | Segmentation of degraded images (A, A*, D) | MedVAE denoising does not help (A* ≤ A); E07 cost A its robustness | measured, redone in E17 |
| E13 | B | FiLM redone (256×256, identical training, controls on c) | fine-tuning +7.4 dB; FiLM ≈ baseline; score C unusable (almost constant in training) | redone |
| E14 | C | JEPA redone (test excluded, probe on the trained layer) | JEPA degrades the representation (−0.056) and the reconstruction; no contraction | redone, refined by E19 |
| E15 | D | Augmentation with realistic degradations (Poisson, JPEG, blur) on A | robustness restored (+0.1 to +0.25 at strong degradations), −0.007 on clean images | option (off by default), redone in E17 |
| E16 | D | Compression controls: same head on the image reduced to 128×128 (R); U-Net of A on the image reduced then enlarged | R 0.405 ≈ B 0.411 (+0.006, not established): the latent does no better than a reduction; in pixels, MedVAE costs −0.004 against −0.039 | controls added |
| E17 | D | E10, E12 and E15 rerun (augmentations different per worker and per epoch, frozen commit) | A 0.428, B 0.411, C 0.410, D 0.427; C ≈ B (−0.001); D ≈ A (0.000); degradation augmentation: +0.010 on clean images | replaces E10, E12, E15 |
| E18 | B | FiLM with scores A and B on 3 seeds | the network uses A and B, but < 0.04 dB (max +0.033); FiLM − baseline not established | conclusion of part B confirmed |
| E19 | C | JEPA on 3 pre-trainings (noise and seeds fixed, probe learning rate chosen on validation) | JEPA lowers the probe of its stage 1 (−0.025, 3 seeds out of 3); stage 2 ≈ MedVAE (+0.019, not established); rank: no net effect | replaces E14 |
| E20 | A | Visibility of thin and thick vessels: CNR (contrast / local background noise) | under Poisson noise, MedVAE improves the CNR (thick +0.16 to +0.54, thin +0.01 to +0.15) but less than 5×5 filters, thin vessels included; without noise, MedVAE changes almost nothing | measured |

Parts: A robustness, B FiLM conditioning, C JEPA, D segmentation (including conditions A, A*, B,
C and D, described in the README). Segmentation effects are test-set Dice scores.

## Protocol

1. **Before**: the reference run is the one of the current state of the project. If it does not
   exist under the current conditions, it is run first.
2. **Change**: the change is committed, then the run is launched (a run launched with
   uncommitted changes is flagged `git_dirty` in its `meta.json`, and the `git diff` is saved in
   `git_diff.patch`). Since the audit (E16 onwards), Slurm jobs are submitted from a worktree frozen
   at the tested commit (`~/MEDVAE-X_runs`): a queued job thus runs exactly that commit, even if the
   repository changes in the meantime.
3. **After**: compare, then write the entry below with its conclusion. **The decision to keep a
   change is made on the validation set (seg_val), not on the test set**: the test set only serves
   to report the result. Up to E15, decisions relied on the test Dice (a flaw found by the audit of
   2026-09-29); the conclusions of E03, E04, E07 and E08 still hold on the validation set (see
   E17), but their test values are no longer fully independent estimates.
4. **If the change is kept**: `python scripts/promote_run.py <run>` updates the official results
   (`finetune/results/`), then the README and figures.

Each run lives in `experiments/runs/<date>_<name>/`: `config.yaml` (exact config), `meta.json`
(commit, GPU, Slurm job), history and scores. The `.pth` checkpoints stay there but are not
versioned.

Launching a run with a modified value, without touching the yaml files:

```bash
python -m finetune.train --config finetune/configs/condition_a.yaml \
  --set training.learning_rate=3e-4 --run-name lr3e-4_condition_a
```

**Metrics (segmentation, 300-image test set).** Per-class Dice and IoU, computed over the whole
test set, then averaged over the classes present (in the prediction or the ground truth),
background included. Dice ≥ IoU always. `scripts/summarize_runs.py` also shows the best
validation Dice (`val_dice`), on which decisions are based.

**Confidence intervals.** A CI computed over images only holds for variants of a single model
(e.g. true c against shuffled c). To compare separately trained models, the CI is over seeds
(Student's t on the per-seed differences); with a single seed, there is no CI.

### Entry template

```markdown
## EXX — Short title (YYYY-MM-DD)

**Hypothesis**: what is expected to improve, and why.
**Change**: what changes (commit `abc1234`).
**Runs**: before `experiments/runs/...` — after `experiments/runs/...`

| | Dice | IoU |
|---|---|---|
| Before | | |
| After | | |

**Conclusion**: kept / dropped, and what to take from it.
```

---

## E01 — Fixing the IoU computation (2026-09-25)

**Observation**: for conditions A and D, the reported IoU was higher than the Dice, which is
impossible (Dice ≥ IoU for any class).

**Cause**: `torchmetrics.segmentation.MeanIoU` expects one-hot masks by default; the project
passed it class-index masks, which it misread without raising an error. On a test case where the
model predicts "background" everywhere, it returned 0.000 instead of 0.283.

**Change** (commit `93390f7`): `MeanIoU` replaced by `MulticlassJaccardIndex(average="macro")`,
computed exactly like the Dice. Regression test: `tests/test_seg_metrics.py` (fails on the old
code, passes on the new one).

**Impact**: the Dice is unchanged (its computation was correct). **Every IoU published so far is
wrong**; since the original checkpoints are not available, they cannot be recomputed; the correct
IoUs are those of E02.

**Conclusion**: kept (bug fix).

## E02 — Reproducing the current state on RTX 3090 (2026-09-25)

**Goal**: obtain a starting point measured under the conditions of this revision. The original
results were obtained on H100/A100, with other library versions and a copy of ARCADE of unknown
provenance; this revision uses RTX 3090 GPUs, torch 2.14 and ARCADE from Zenodo.

**Code**: commit `10f4048`, configs unchanged.
**Runs**: `experiments/runs/2026-09-25_*_e02_*`

**Adaptations to the 3090 (24 GB), with no effect on the computation**: MedVAE fine-tuning
processes each batch of 4 as 4 micro-batches of 1 with gradient accumulation, and condition D
passes the images through the frozen MedVAE in groups of 2. In both cases the result is
mathematically identical (mean L1 loss, per-image normalisations).

| Condition | Original Dice | E02 Dice | E02 IoU (fixed) | Epochs |
|---|---|---|---|---|
| A — U-Net | 0.433 | **0.436** | 0.330 | 100 |
| A* — U-Net A on reconstructed images | 0.435 | **0.437** | 0.329 | — (eval.) |
| B — MedVAE latent + head | 0.038 | **0.039** | 0.039 | 22 (early stop) |
| C — fine-tuned MedVAE latent + head | 0.039 | **0.039** | 0.039 | 22 (early stop) |
| D — MedVAE → U-Net | 0.451 | **0.445** | 0.335 | 100 |

MedVAE fine-tuning (prerequisite of C): validation L1 loss 0.02617 → 0.02408 (epoch 49/50),
identical to the original (0.02613 → 0.02408, epoch 49/50). *(0.02617 is the value after the
first epoch, not that of the pre-trained MedVAE, which was not measured.)*

**Conclusions**

- **Successful reproduction**: every Dice is within ±0.006 of the original despite the change of
  GPU, libraries and dataset source. E02 is the "before" reference for the following experiments.
- **The true IoUs are ~0.33** for A, A* and D (against ~0.50 reported, see E01).
- **B and C detect no artery**: Dice of 0.98 on the background and < 0.01 on the 25 artery
  classes, no progress on validation, stopped at epoch 22. They predict "background" everywhere.
  The project's conclusion at the time ("the latent is not suited to dense prediction") therefore
  rests on models that learned nothing: whether it is a real limit of the latent or a training
  problem remains to be established.
- **D ≈ A** (0.445 vs 0.436): the A/D gap is of the same order as the original/reproduction
  variation (0.006), so it is not significant on a single run.

## E03 — Head learning rate of B and C: 5e-5 → 1e-3 (2026-09-26)

**Preliminary diagnosis** (`experiments/diagnostics/diag_latent_b.py`, commit `b7854d8`): the
MedVAE latent is not the cause. The noise of the random sampling is negligible (standard deviation
0.001 against 7 for the signal) and the latent holds as much information on the vessels as an
image reduced to the same size. On 8 images, however, the head of B learns almost nothing with its
learning rate (Dice 0.05), but reaches Dice 0.72 and 13 arteries with lr = 1e-3.

**Hypothesis**: B and C stay stuck on "everything is background" because their learning rate
(5e-5) is too low for a head trained from scratch.
**Change**: `training.learning_rate` 5e-5 → 1e-3, nothing else (commit `247793b`,
`--set training.learning_rate=1e-3`). C uses the fine-tuned MedVAE of E02.
**Runs**: before `*_e02_condition_{b,c}` — after `2026-09-26_025705_e03_lr1e-3_condition_{b,c}`

| | Dice | IoU | Arteries detected (Dice > 0.01) | Epochs |
|---|---|---|---|---|
| B before (E02) | 0.039 | 0.039 | 0 / 25 | 22 (early stop) |
| **B after** | **0.097** | **0.070** | **12 / 25** | 97 (early stop) |
| C before (E02) | 0.039 | 0.039 | 0 / 25 | 22 (early stop) |
| **C after** | **0.091** | **0.066** | **12 / 25** | 100 |

**Conclusion**: kept (learning rate 1e-3 in the configs of B and C, results promoted). B and C
finally learn (Dice ×2.5, 12 arteries detected among the largest, Dice up to 0.25 per class), but
remain far below A (0.436).

- **Underfitting**: training Dice (~0.09) ≈ validation Dice (~0.11); the model cannot do better
  even on the training images. The limit is the capacity of the head (no convolution at the latent
  resolution, very small receptive field), not overfitting.
- **C ≈ B**: fine-tuning MedVAE on ARCADE brings nothing here (gap 0.006, of the same order as the
  run-to-run variation).
- The original conclusion ("the latent is not suited to dense prediction") is not established: a
  head able to use the context must be tested first.

## E04 — U-Net head at the latent resolution for B and C (2026-09-26)

**Hypothesis**: after E03, B and C underfit (training Dice ≈ validation ≈ 0.09). The `SegHead`
head has no convolution at the latent resolution: each pixel only sees its immediate
neighbourhood, whereas identifying an artery segment (LAD, circumflex…) requires seeing a large
part of the image.
**Change**: new `LatentUNetHead` head (commit `388ece7`): a small U-Net on the latent
(128 → 64 → 32 → 16 → 128, skip connections), then the same ×4 upsampling to 512×512;
`base_channels=64`, 7.8 M parameters (U-Net of A: 24 M). Nothing else changes (same latent, loss,
lr 1e-3). Launched with `--set model.architecture=latent_unet_head --set model.base_channels=64`.
Preliminary test on 8 images (lr 1e-3, 1000 steps): Dice 0.995 against 0.749 for `SegHead`.
**Runs**: before `*_e03_lr1e-3_condition_{b,c}` — after `2026-09-26_114934_e04_latent_unet_head_condition_{b,c}`

| | Dice | IoU | Arteries detected | Train / val Dice (best) |
|---|---|---|---|---|
| B before (E03) | 0.097 | 0.070 | 12 / 25 | 0.090 / 0.109 |
| **B after** | **0.409** | **0.304** | **21 / 25** | 0.452 / 0.444 |
| C before (E03) | 0.091 | 0.066 | 12 / 25 | 0.092 / 0.107 |
| **C after** | **0.398** | **0.292** | **22 / 25** | 0.466 / 0.446 |
| *Reference A (U-Net on the image)* | *0.436* | *0.330* | *22 / 25* | *0.650 / 0.507* |
| *Reference D (MedVAE → U-Net)* | *0.445* | *0.335* | *23 / 25* | *0.567 / 0.497* |

**Conclusion**: kept (`latent_unet_head`, `base_channels: 64` in the configs of B and C, results
promoted). Dice ×4 for B and C, which reach 94 % of the Dice of A (0.409 against 0.436) starting
only from the MedVAE latent, compressed ×16.

- **The original conclusion of the project is reversed**: the MedVAE latent *is* usable for dense
  coronary segmentation; the failure came from the training (E03) and the head (E04), not from the
  latent.
- **C ≈ B** once again (0.398 against 0.409): fine-tuning MedVAE on ARCADE does not help.
- **Still underfitting** (train ≈ val ≈ 0.45, against 0.65 / 0.51 for A) and the curves plateau
  when the learning rate reaches its minimum (100 epochs, best score at epochs 90-97): a wider head
  or a longer training could narrow the gap with A further.

## E05 — Adoption of E04 (2026-09-26)

`LatentUNetHead` (`base_channels: 64`) becomes the head of B and C in the configs (commit
`e361732`) and the E04 runs are promoted to official results.

## E06 — Seed-to-seed variability (2026-09-26)

**Goal**: so far each condition had a single run; without knowing the run-to-run variability, one
cannot tell whether a gap is real.
**Change**: none; seeds 43 and 44 for A, B, C and D (seed 42 = runs E02 for A and D, E04 for B and
C, same configuration). The seed also sets the training/validation split (80/20 of seg_train): the
measured variability includes this effect.
**Runs**: `2026-09-26_*_e06_seed4{3,4}_condition_{a,b,c,d}` (commit `e361732`)

| Condition | Dice | Artery Dice | Artery IoU |
|---|---|---|---|
| A — U-Net | 0.424 ± 0.010 | 0.400 ± 0.011 | 0.294 ± 0.007 |
| B — MedVAE latent + U-Net head | 0.409 ± 0.009 | 0.385 ± 0.009 | 0.276 ± 0.005 |
| C — fine-tuned MedVAE latent + U-Net head | 0.397 ± 0.005 | 0.372 ± 0.005 | 0.264 ± 0.004 |
| D — MedVAE → U-Net | 0.441 ± 0.003 | 0.418 ± 0.003 | 0.305 ± 0.003 |

(mean ± standard deviation over 3 seeds; "artery" = mean over the 25 artery classes, background
excluded, see E09; table produced by `scripts/summarize_runs.py`)

**Conclusions**

- **The seed-to-seed variability is about ±0.01 Dice** (up to 0.02 between two seeds for A): no
  smaller gap should be read into a single run.
- **D beats A for all 3 seeds** (+0.009, +0.020, +0.024; +0.018 on average): the advantage of D
  reported originally holds, although it could not be shown on a single run (E02).
- **B < A** (−0.015 on average) and **C < B** for 2 seeds out of 3 (seed 43: +0.0004, a tie): the
  MedVAE latent alone stays slightly below the full image, and fine-tuning MedVAE on ARCADE degrades
  it slightly. *(Wording corrected after the audit; see E17: no effect of the fine-tuning.)*

## E07 — Strength of the Gaussian augmentation noise (2026-09-26)

**Observation**: `A.GaussNoise(p=0.3)` without parameters adds, with albumentations ≥ 2 (2.0.8
installed), a noise with a standard deviation of 20 to 44 % of the maximum (~65-84 levels out of
255): 30 % of the training images are almost destroyed. With albumentations 1.x, the same line gave
a noise with a standard deviation of ~3-7 levels, presumably the original intent.
**Change**: option `data.gauss_noise_std_range` (commit `6dca6a2`), tested with `[0.0124, 0.0277]`
(standard deviation 3.2-7.1 / 255, the equivalent of the old default) on A.
**Runs**: before = A of E06 — after `2026-09-26_*_e07_light_noise_seed4{2,3,4}_condition_a`

| A | Dice | Artery Dice | Artery IoU |
|---|---|---|---|
| Before (E06) | 0.424 ± 0.010 | 0.400 ± 0.011 | 0.294 ± 0.007 |
| Light noise | 0.433 ± 0.008 | 0.410 ± 0.008 | 0.294 ± 0.006 |

Paired gap per seed: +0.004, +0.007, +0.018 (+0.010 on average).

**Conclusion**: kept. The gain is positive for all 3 seeds but of the same order as the
variability, and zero in IoU: the effect is small. The change is kept mainly because it restores
the augmentation originally intended. *Added after E12: this gain is paid for in robustness to
degradations (Dice 0.34 → 0.02 at the strongest Poisson noise), see E12 and E15.*

## E08 — Validation on the official seg_val (2026-09-26)

**Observation**: ARCADE provides an official validation set (`seg_val`, 200 annotated images) that
was never used: validation was taken from 20 % of `seg_train`, and the model trained on 800 images
instead of 1000.
**Change**: option `data.validation: official` (commit `6dca6a2`): training on the 1000 images of
`seg_train`, model selection on `seg_val`. The seed no longer changes the split, only the
initialisation and the batch order. Tested on A.
**Runs**: before = A of E06 — after `2026-09-26_*_e08_official_val_seed4{2,3,4}_condition_a`

| A | Dice | Artery Dice | Artery IoU |
|---|---|---|---|
| Before (E06) | 0.424 ± 0.010 | 0.400 ± 0.011 | 0.294 ± 0.007 |
| Official validation | **0.454 ± 0.018** | **0.431 ± 0.019** | **0.316 ± 0.011** |

Paired gap per seed: +0.028, +0.015, +0.048 (+0.030 on average).

**Conclusion**: kept. Clear gain, positive for all 3 seeds, on every metric.

## E09 — Metrics on the arteries only (2026-09-26)

**Change**: `SegMetrics` also computes `dice_fg_mean` and `iou_fg_mean`, averaged over the 25
artery classes present (predicted or true), background excluded, as in the ARCADE challenge (commit
`f3c156c`). `finetune/evaluate.py` re-evaluates a finished run with the current metrics; all runs
E02 to E06 were re-evaluated (their `results.json` were rewritten with the code of commit `f3c156c`,
with no trace in `meta.json`) (Dice identical to within 1e-4; D varies by 2e-4 because MedVAE
samples its latent at random). Class 12 is absent from the test set; category 26 ("stenosis") never
appears in the segmentation annotations, so the 26 classes of the model (background + 25 segments)
are correct.

**Effect**: the easy background (Dice ~0.99) inflated the mean: the artery Dice is ~0.024 below
`dice_mean` for every condition (e.g. A: 0.424 → 0.400). The ranking of the conditions is unchanged.

## E10 — New reference state: E07 + E08 on A, B, C, D (2026-09-28)

**Change**: the two changes validated on A (validation on seg_val, E08; light augmentation noise,
E07) applied to the configs of A to D (commit `f3b6538`), 3 seeds per condition; A* = U-Net A of
seed 42 applied to the MedVAE reconstructions.
**Runs**: before = E06 (same seeds) — after `2026-09-2{7,8}_*_e10_seed4{2,3,4}_condition_{a,b,c,d}`
and `2026-09-28_*_e10_seed42_condition_astar`

| Condition | Dice before (E06) | Dice after (E10) | Artery Dice | Artery IoU | Gap per seed (42 / 43 / 44) |
|---|---|---|---|---|---|
| A — U-Net | 0.424 ± 0.010 | **0.452 ± 0.005** | 0.430 ± 0.005 | 0.308 ± 0.002 | +0.018 / +0.028 / +0.039 |
| A* — U-Net A on reconstructions | — | 0.448 (seed 42) | 0.425 | 0.303 | −0.006 relative to A |
| B — MedVAE latent | 0.409 ± 0.009 | **0.417 ± 0.008** | 0.393 ± 0.008 | 0.277 ± 0.005 | +0.017 / +0.014 / −0.007 |
| C — fine-tuned MedVAE latent | 0.397 ± 0.005 | **0.427 ± 0.002** | 0.403 ± 0.002 | 0.285 ± 0.002 | +0.028 / +0.028 / +0.035 |
| D — MedVAE → U-Net | 0.441 ± 0.003 | **0.455 ± 0.004** | 0.432 ± 0.004 | 0.311 ± 0.003 | +0.015 / +0.015 / +0.011 |

**Conclusions**

- **Every condition improves**, A and C the most (+0.03); B the least (+0.008, one seed down).
  Results promoted (seed 42 of each condition, as before).
- **D ≈ A**: +0.003 on average (+0.005, +0.006, −0.004), within the variability. The advantage of
  D measured in E06 (+0.018, 3 seeds out of 3) almost vanishes. Hypothesis: with the too-strong
  augmentation noise before E07, 30 % of the training images of A were almost destroyed, whereas D
  saw them after MedVAE, which denoises (E11); D was therefore less penalised by this noise. A also
  gained twice as much as D (+0.028 against +0.013).
- **C > B for 2 seeds out of 3, a tie at seed 42** (+0.010 on average), the opposite of E06:
  fine-tuning MedVAE on ARCADE may help segmentation from the latent slightly, but the gap is of the
  order of the variability. *(Wording corrected after the audit; E17 does not confirm this effect:
  C − B = −0.001.)*
- **A* ≈ A** (−0.006 on one seed): compression alone costs little.
- Segmentation from the latent (B, C) stays ~0.03 below A (92-94 % of its Dice).

## E11 — Robustness of MedVAE to degradations, corrected measurement (2026-09-27)

**Findings on the original study** (part A, `medvae_eval/scripts/run_maskedPSNR_sweep.py`):
- the main metric compares the reconstruction with the **degraded** input: since a blurred image is
  easier to reproduce, "blurred images reconstruct better" is almost tautological and says nothing
  about robustness;
- images are mapped to [-1, 1] by a per-image min-max (which changes with the degradation): the
  three metrics are not on the same scale;
- the latent is sampled at random, the noise is not seeded, and only 50 seg_train images are used.

**New protocol** (`medvae_eval/robustness/sweep.py`, commit `34b8619`): deterministic latent
(posterior mean), fixed [0, 1] scale, noise seeded per image and per level, 100 seg_val images, 3
degradations × 10 levels (same ranges as originally), 95 % bootstrap CIs over images. The question
asked is **Δ = PSNR(clean, reconstruction) − PSNR(clean, degraded input)**: Δ > 0 means that MedVAE
brings the image closer to its clean version.

**Results** (`experiments/robustness/e11_sweep/`, figures `robustness_*.png`): MedVAE reconstructs
the clean image at 33.1 dB (vessels: 33.1 dB), which sets a ceiling.

| Degradation | Image Δ PSNR (dB), weakest to strongest level | Vessel Δ (dB) | Images restored |
|---|---|---|---|
| Poisson noise | **+1.2 → +6.6** | +0.9 → +6.8 | 99-100 % |
| JPEG (quality 95 → 5) | −11.7 → −0.5 | −11.3 → −0.1 | 0-9 % |
| Gaussian blur (kernel 5 → 31) | −2.0 → +0.08 | −2.8 → −0.2 | 3-82 % |

(95 % CI width ≤ 0.53 dB everywhere)

**Conclusions**

- **MedVAE reduces Poisson noise**, more so as the noise gets stronger (+6.6 dB at the noisiest
  level); see below the "simple filter" control, which does clearly better.
- **It restores neither JPEG nor blur.** For JPEG, its own reconstruction error (ceiling at 33 dB)
  exceeds the compression artefacts: it always degrades the image. For blur, it recovers no detail;
  the reconstruction simply follows the blurred input (Δ ≈ 0 on the image, slightly negative on the
  vessels).
- *Added after the audit (2026-09-29): "simple filter" control* (`medvae_eval/robustness/baselines.py`,
  `experiments/robustness/e11_baselines/`). On the same 100 images and the same seeded noise, a 5×5
  Gaussian blur gains **+6.3 → +10.3 dB** (vessels +7.1 → +10.9) and a 5×5 median filter
  **+7.2 → +10.0 dB**, against +1.2 → +6.6 dB for MedVAE: MedVAE reduces Poisson noise, but 3.4 to
  6.3 dB less than a trivial filter; "a true denoiser" was an overstatement. On the strongest JPEG,
  the median filter even gains a little (+0.4 dB), MedVAE never does.
- The original result on blur ("blurred images reconstruct better", r ≈ −0.998) came from the
  metric: the orange curve of the figures (reconstruction vs degraded input) rises with the blur,
  whereas the blue curve (vs clean image) falls.

## E12 — Segmentation of degraded images: does MedVAE make segmentation more robust? (2026-09-29)

**Question**: E11 shows that MedVAE removes some Poisson noise (+1.2 to +6.6 dB). Does this help to
segment degraded images?
**Protocol** (`medvae_eval/robustness/downstream.py`, commits `dccc438` and `91e77bf`): the 300 test
images are degraded (Poisson, JPEG, blur; same models and ranges as E11, levels 0, 3, 6, 9 out of
10, seeded noise, same degraded images for every model), then segmented by A, A* (MedVAE then the
U-Net of A) and D, for the 3 seeds of E10. Artery Dice.
**Results**: `experiments/robustness/e12_downstream/` (`main.md`, `main.png`)

| Artery Dice (3 seeds) | A | A* | D |
|---|---|---|---|
| Clean image | 0.429 ± 0.005 | 0.426 ± 0.006 | 0.432 ± 0.004 |
| Poisson, levels 0 / 6 / 9 | 0.400 / 0.345 / 0.025 | 0.379 / 0.307 / 0.040 | 0.400 / 0.353 / 0.079 |
| JPEG, levels 3 / 6 / 9 | 0.426 / 0.414 / 0.051 | 0.421 / 0.413 / 0.087 | 0.429 / 0.425 / 0.113 |
| Blur, levels 3 / 6 / 9 | 0.393 / 0.297 / 0.151 | 0.395 / 0.304 / 0.158 | 0.397 / 0.287 / 0.168 |

- **MedVAE denoising does not help segmentation**: under Poisson noise, A* (images denoised by
  MedVAE) does *worse* than A at low and medium levels (−0.02 to −0.04); the U-Net of A is more
  hampered by the reconstructions of noisy images than by the noise itself.
- **D ≈ A** at all low and medium levels; at the extreme Poisson and JPEG levels, all collapse
  (Dice ≤ 0.11), D slightly less; the strongest blur divides the Dice by two to three (0.15 to
  0.17). *(Wording corrected after the audit: "all < 0.12" was wrong for blur.)*
- JPEG: no effect up to level 6 (quality ≥ 35), collapse at level 9 (quality 5).

**Complement: effect of the augmentation noise (E07) on robustness.** Same protocol on the models
trained with the old, too-strong noise (A and D of E02/E06) and with the corrected noise (A of E07),
all with the old 80/20 split (`experiments/robustness/e12_aug_noise/`).

| Artery Dice of A (3 seeds) | Strong noise (E06) | Light noise (E07) |
|---|---|---|
| Clean image | 0.400 ± 0.011 | 0.410 ± 0.008 |
| Poisson, levels 6 / 9 | 0.379 / **0.340** | 0.341 / **0.021** |
| JPEG, level 9 | 0.131 | 0.029 |
| Blur, levels 6 / 9 | 0.325 / 0.253 | 0.283 / 0.166 |

- **E07 has a hidden cost**: the unintended, too-strong augmentation noise made A very robust to
  degradations (Dice 0.34 at the strongest Poisson noise, against 0.02 with the corrected noise).
  The +0.010 gain on clean images was paid for in robustness.
- Hence E15: can an augmentation with realistic degradations recover this robustness without
  losing the gain on clean images?

## E13 — Part B redone: FiLM conditioning on a quality score (2026-09-27)

**Findings on the original study** (notebooks 04 to 07 of `medvae_eval/cvae/`):
- training and evaluation at **64×64**: at this resolution, the pre-trained MedVAE
  (`vae_4x_4c_2D`) only reconstructs at **20.7 dB** on seg_val, against **34.8 dB** at 256×256 (runs
  `e13_film_pretrained_64px` and `e13_film_pretrained`), and the vessels are ~1 pixel wide;
- loss = mean MSE + 1e-6 × KL **summed** over the latent: the KL term is not on the same scale as
  the MSE (at 256×256, it weighs ~700 times the MSE and makes the validation error rise during
  training). The "internal ELBO loss" on which the cVAE gained 31 to 45 % therefore mostly measures
  the KL, not the reconstruction;
- the cVAE (2500 steps of FiLM alone, then 5000 full steps) and the baseline (7500 full steps) are
  not trained the same way;
- a single run per model, 60 test images: the ablation compares two separately trained models (true
  c against c = 0.5), so the gap mixes the use of c with the run-to-run variability, which the
  Wilcoxon test over images does not cover.

**New protocol** (`medvae_eval/film/train_film.py`, commits `34b8619` and `da55a67`):
- 256×256; seg_train split 900 / 100 (fixed split); **test = seg_val** (200 images, never seen for
  training nor for selection);
- baseline and FiLM trained **identically**: all parameters, 3000 steps, AdamW lr 1e-5 cosine,
  batch 4, loss = MSE + 1e-6 × KL / number of pixels (MedVAE / latent-diffusion convention);
  selection on the validation MSE; 3 seeds;
- FiLM: 19 blocks, heads initialised at zero, +1.70 M parameters (+3.1 % of the 55.3 M of
  `vae_4x_4c_2D`);
- **inference-time controls, on the same model**: shuffled c (fixed permutation of the scores
  among the 200 images) and constant c (training mean). If the model uses c, the true c must beat
  both;
- score c of approach C (learned), the one for which the original study concluded that c is used.

**Runs**: `2026-09-27_*_e13_film_{pretrained,baseline_seed4?,C_seed4?}`; analysis
`python medvae_eval/film/analyze.py` (command in the script) → `experiments/film/e13/`.

| Model (test seg_val, 256×256) | PSNR (dB) | Vessel PSNR (dB) | SSIM | HaarPSI |
|---|---|---|---|---|
| Pre-trained MedVAE | 34.81 | 35.57 | 0.949 | 0.939 |
| Fine-tuned baseline (no c) | 42.261 ± 0.013 | 40.891 ± 0.005 | 0.970 | 0.983 |
| FiLM, true c | 42.287 ± 0.018 | 40.915 ± 0.009 | 0.970 | 0.983 |
| FiLM, shuffled c | 42.285 ± 0.017 | 40.914 ± 0.009 | 0.970 | 0.983 |
| FiLM, constant c | 42.300 ± 0.021 | 40.925 ± 0.010 | 0.970 | 0.984 |

| Paired difference (PSNR, dB) | Mean | 95 % CI (images) | Per seed (42 / 43 / 44) |
|---|---|---|---|
| Baseline − pre-trained | **+7.45** | [+7.06, +7.85] | +7.44 / +7.46 / +7.44 |
| FiLM (true c) − baseline | +0.026 | [+0.015, +0.038] (over images, see note) | +0.055 / +0.004 / +0.019 |
| FiLM: true c − shuffled c | +0.002 | [−0.007, +0.010] | +0.002 / +0.002 / +0.001 |
| FiLM: true c − constant c | −0.013 | [−0.021, −0.006] | −0.015 / −0.015 / −0.009 |

(mean ± standard deviation over 3 seeds; same conclusions on the vessel PSNR, the SSIM and the
HaarPSI, see `experiments/film/e13/paired.csv`)

**Conclusions**

- **Fine-tuning MedVAE on ARCADE greatly improves the reconstruction**: +7.4 dB PSNR (+5.3 dB on
  the vessels) over the pre-trained model, for all 3 seeds.
- **FiLM does neither better nor worse than the baseline**: +0.03 dB, positive for all 3 seeds but
  very small (< 0.1 dB). The original result (cVAE worse by 21 to 49 % in MSE) does not reproduce
  when both models are trained identically.
- **The model does not use the quality information**: replacing the true c by that of another
  image changes nothing (+0.002 dB, CI containing 0), and a constant c even does very slightly
  better. The learned FiLM modulations do depend on c (|γ(1) − γ(0)| up to 0.05), but this
  dependence does not help the reconstruction: the small gain over the baseline comes from the added
  parameters (one affine modulation per channel), not from the score. The original conclusion
  "approach C uses c" is not confirmed.
- *Notes after the audit (2026-09-29)*: (1) the "FiLM − baseline" CI above is over images, whereas
  it compares separately trained models; over the 3 seeds, the 95 % CI (Student) is
  **[−0.04, +0.09]**: the gap is not established. (2) Score C could not be used: on the 900 training
  images, it averages 0.92, with 79 % above 0.9 (5-95 % range [0.70, 1.00]), against 0.73 on average
  on the test set (26 % below 0.5). The model almost never saw a low score, and over the range it
  saw, γ varies by at most 0.02. Hence E18: scores A and B on 3 seeds.

**Additional checks (seed 42)**: scores of approaches A (weighted) and B (PCA), and a learning rate
100 times higher for the FiLM layers (`--film-lr 1e-3`, commit `8445406`), to rule out that c is
ignored because the FiLM heads learn too slowly. Runs `*_e13_film_{A,B,C_filmlr1e-3}_seed42`;
analysis `experiments/film/e13_seed42_checks/`.

| Seed 42, PSNR (dB) | FiLM − baseline | true c − shuffled c | true c − constant c |
|---|---|---|---|
| Score C (reference above) | +0.055 | +0.002 [−0.007, +0.010] | −0.015 |
| Score C, FiLM lr 1e-3 | **+0.239** [+0.16, +0.33] | +0.011 [−0.008, +0.031] | −0.001 [−0.018, +0.018] |
| Score A | +0.067 [+0.05, +0.09] | +0.009 [+0.003, +0.016] | +0.006 [+0.002, +0.011] |
| Score B | +0.078 [+0.06, +0.10] | **+0.037** [+0.020, +0.054] | +0.023 [+0.010, +0.036] |

- **A higher FiLM lr improves FiLM (+0.24 dB), but not thanks to c**: the constant c does as well
  as the true one. The FiLM layers act as additional parameters.
- **With scores A and B, the true c beats both controls** (CI excluding 0): the network makes some
  use of these scores, built from sharpness and contrast metrics computed over the whole image (a
  global information, whereas this MedVAE only has convolutions). The effect remains tiny:
  ≤ 0.04 dB on the image, ≤ 0.012 dB on the vessels, a single seed.
- Overall conclusion: **quality conditioning brings no useful gain to the reconstruction**. Score C
  (learned) is not used; A and B are, very slightly. This is the opposite of the original study (C
  used, A reversed), which compared separately trained models on a single run.

## E14 — Part C redone: JEPA at stage 2, corrected evaluation (2026-09-29)

**Findings on the original study** (`jepa_adaptation/`, notebook `pretraining_analysis.ipynb`):
- the ARCADE test images were part of the pre-training (fixed: 1080 / 120 images from seg_train +
  seg_val, test excluded; commit `dfca730`);
- **"stage 1" is not MedVAE**: it is an autoencoder of the same family but smaller (`ch` 64,
  `ch_mult` [1, 2, 4, 4]: factor 8 per side, 64×64×1 latent, 21 M parameters), trained **from
  scratch** on ARCADE (15 epochs, reconstruction, perceptual and adversarial losses). The comparison
  "MedVAE → stage 1 → stage 2" therefore mixes architecture, compression rate (×4 against ×8 per
  side) and data;
- **the latent-structure measurements were made on a layer that is never trained**: the output
  `channel_proj(channel_ds(z) + z)`, ignored by the stage-1 loss and off the path of the JEPA loss
  (which acts on the features before `conv_out`). The "90 % of the variance in 25 dimensions"
  described a random projection of the latent;
- the linear probe predicted the dominant artery class of each image, on 60 images.

**New evaluation** (commits `dfca730`, `a1cf08b`, `8f4086f`), on the 300 test images, with the
models retrained without the test set: reconstruction (`eval/reconstruction.py`), per-pixel linear
probe vessel / background on the layer trained by JEPA (`eval/probe.py`: standardised features, 1×1
convolution, training on seg_train, selection on seg_val, 3 seeds), dimensionality of this layer
(`eval/feature_rank.py`). Results: `experiments/jepa/`.

| | MedVAE (×4, pre-trained) | Stage 1 (×8, from scratch) | Stage 2 JEPA |
|---|---|---|---|
| Reconstruction: PSNR / vessels (dB) | 34.8 / 35.5 | 30.2 / 30.9 | **17.8** / 24.5 |
| Vessel linear probe, Dice (lr 1e-2, 100 epochs) | 0.517 ± 0.005 | **0.541 ± 0.005** | 0.485 ± 0.006 |
| Effective rank of the features (channels) | 251 (512) | 111 (256) | 139 (256) |
| Components for 90 % of the variance | 61 | 22 | 34 |

- **The probe must converge**: at 20 epochs (lr 1e-3), the scores were 0.39 / 0.43 / 0.39 and the
  best epoch was the last one *(for MedVAE and stage 1; for stage 2: 16, 18, 16)*; at 60 epochs
  0.45 / 0.50 / 0.45. *(Correction, see E19: I had written "the ranking is the same in all three
  settings"; this is wrong, stage 2 equals MedVAE at 20 and 60 epochs.)* The probe on the 26
  segments (Dice ~0.03) and on the 1-channel latent (~0.03) separates nothing: a per-pixel linear
  classifier cannot name a segment, nor separate the vessels with a single channel.
- **JEPA degrades the representation it sets out to improve**: −0.056 Dice relative to stage 1, and
  below the original MedVAE *(on a single pre-training; E19, on 3, confirms the drop relative to
  stage 1 but not the gap to MedVAE)*; the reconstruction collapses (frozen decoder).
- **No dimensional contraction**: on the trained layer, JEPA *increases* the effective rank
  (111 → 139) and the number of components (22 → 34), which is the expected effect of the SIGReg
  regularisation *(an increase that does not reproduce over 3 pre-trainings, see E19)*. The original
  conclusion ("a more compact and more discriminative latent") is not confirmed.

## E15 — Augmentation with realistic degradations (2026-09-29)

**Hypothesis**: E12 shows that A, trained with the corrected augmentation noise (E07), collapses on
strongly degraded images. Showing the model acquisition degradations during training should restore
this robustness.
**Change**: option `data.degradation_aug_p` (commit `91e77bf`): with probability 0.3, one
degradation among Poisson noise (scale 0.05-1), JPEG (quality 5-95) and Gaussian blur (kernel 3-31).
Tested on A, nothing else changes (E10 config, 3 seeds, `--set data.degradation_aug_p=0.3`).
**Caution**: these are the same families and ranges as the robustness evaluation (E12), which is
therefore "in distribution" for the augmentation.
**Runs**: before = A of E10 — after `2026-09-29_*_e15_degradation_aug_seed4{2,3,4}_condition_a`;
robustness `experiments/robustness/e15_downstream/` (`comparison.md`, `comparison.png`)

| Artery Dice (3 seeds) | A (E10) | A + degradations (E15) |
|---|---|---|
| Clean image | 0.430 ± 0.005 | 0.423 ± 0.011 |
| Poisson, levels 6 / 9 | 0.345 / 0.025 | **0.400 / 0.266** |
| JPEG, level 9 | 0.051 | **0.289** |
| Blur, levels 6 / 9 | 0.297 / 0.151 | **0.403 / 0.373** |

Paired gap on clean images, per seed: +0.002, −0.012, −0.011 (−0.007 on average; artery IoU
−0.002).

**Conclusion**: robustness is largely restored (gains of 0.1 to 0.25 Dice at strong degradations),
at a cost on clean images of the order of the seed-to-seed variability (2 seeds out of 3 down).
Option kept but **off by default**: the official results remain those of E10 (clean images), and
turning it on would require retraining B, C and D to keep the conditions comparable. To be decided
according to the intended use (robustness or score on clean images).

## E16 — Compression controls for segmentation (2026-09-29)

**Why** (audit, MAJ-12): the conclusion "compression preserves the information needed for
segmentation" had no control. B and C (128×128×1 latent, 16 times fewer values than the image) were
only compared with A (full image, a U-Net three times larger); A* was compared with no trivial
compression.
**Controls**:
- **R** (commit `4b6cf64`, `finetune/configs/condition_r.yaml`): the head of B, same architecture and
  same training, applied to the image averaged over 4×4 blocks (128×128×1, as many values as the
  latent). 3 seeds, from the frozen worktree;
- **A↓↑** (commit `04a8e11`, `finetune/eval_resampled.py`): the U-Net of A, without retraining,
  applied to the image reduced to 128×128 then enlarged back to 512×512 (bilinear or bicubic). This
  is the trivial compression that corresponds to A*.
**Runs**: `2026-09-29_*_e16_seed4?_condition_r`; `experiments/diagnostics/e16_resampled.csv`
(A models of E17).

| Artery Dice (test, 3 seeds) | Mean | Paired gap (95 % CI over seeds) | Per seed |
|---|---|---|---|
| B — MedVAE latent + head | 0.411 ± 0.005 | | |
| C — fine-tuned latent + head | 0.410 ± 0.009 | | |
| **R — image reduced to 128×128 + same head** | 0.405 ± 0.002 | B − R = +0.006 [−0.002, +0.014] | +0.009 / +0.003 / +0.007 |
| | | (validation: B − R = +0.005 [−0.020, +0.030]) | +0.016 / −0.002 / +0.001 |
| A* — U-Net A on the MedVAE reconstruction | 0.424 ± 0.008 | A* − A = −0.004 [−0.009, +0.002] | −0.003 / −0.002 / −0.006 |
| **A↓↑ — U-Net A on the image reduced then enlarged** (bilinear) | 0.389 ± 0.018 | A↓↑ − A = −0.039 [−0.070, −0.008] | −0.049 / −0.042 / −0.025 |

(bicubic: 0.390 ± 0.022; A* − A↓↑ = +0.035 [−0.001, +0.071] in Dice, +0.030 [+0.003, +0.058] in IoU)

**Conclusions**

- **Used as the input of a trained head, the MedVAE latent does hardly better than a trivial
  reduction**: with the same head, the image averaged to 128×128 almost reaches the Dice of the
  latent (B − R = +0.006, in favour of B for all 3 seeds on the test set but with a CI containing 0,
  and not on the validation set). That B and C reach 96 % of the Dice of A mainly shows that a
  128×128 resolution suffices for this task with a suitable head; the latent preserves the
  information useful for segmentation as well as a ×4 reduction, not clearly better.
- **Decoded back to pixels, MedVAE preserves much more than a trivial reduction**: without
  retraining, the U-Net of A loses 0.004 on the MedVAE reconstructions, but 0.039 on the image
  reduced then enlarged (3 seeds out of 3), where thin vessels are erased. From 128×128 values, the
  MedVAE reconstruction restores details that an interpolation does not.
- The two controls answer two different questions: R is trained for its input (compression then
  learning), A↓↑ is not (compression only, like A*).

## E17 — E10, E12 and E15 rerun with corrected augmentations (2026-09-29)

**Why**: the audit found that with albumentations 2, each DataLoader worker received a copy of the
same random generator, without a new seed: the workers drew the same augmentations, and the same
sequence came back at every epoch. This flaw affected every segmentation experiment (E02 to E15).
In addition, the official results of B and D and several published figures came from `git_dirty`
runs, and queued jobs ran the code present when they started.
**Change**: per-worker augmentation seed (`seed_worker_augmentations`, commit `4b6cf64`, tested in
`tests/test_dataset.py`); every run launched from the worktree frozen at this commit (all clean).
**Protocol**: E10 config unchanged, 3 seeds per condition; A* = U-Net of A applied to the
reconstructions (3 seeds through the robustness evaluation, seed 42 through `eval_astar`);
robustness (E12 protocol) and degradation augmentation (E15, p = 0.3) redone on these runs.
Decisions on the validation set.
**Runs**: `2026-09-29_*_e17_seed4?_condition_{a,b,c,d}`, `*_e17_seed42_condition_astar`,
`*_e17_degradation_aug_seed4?_condition_a`; robustness `experiments/robustness/e17_downstream/`
(`report_figure.md`, comparison with E12 and E15 in `vs_e12_e15.md`).

| Condition (3 seeds) | Validation Dice | Artery Dice (test) | Artery IoU (test) | E10, artery Dice |
|---|---|---|---|---|
| A — U-Net | 0.481 ± 0.010 | 0.428 ± 0.008 | 0.309 ± 0.006 | 0.430 ± 0.005 |
| A* — U-Net A on reconstructions | — | 0.424 ± 0.008 | 0.305 ± 0.006 | 0.425 (seed 42) |
| B — MedVAE latent | 0.473 ± 0.009 | 0.411 ± 0.005 | 0.292 ± 0.004 | 0.393 ± 0.008 |
| C — fine-tuned latent | 0.473 ± 0.006 | 0.410 ± 0.009 | 0.291 ± 0.008 | 0.403 ± 0.002 |
| D — MedVAE → U-Net | 0.488 ± 0.010 | 0.427 ± 0.009 | 0.309 ± 0.005 | 0.432 ± 0.004 |
| A + degradations (p = 0.3) | 0.489 ± 0.011 | 0.437 ± 0.010 | 0.316 ± 0.008 | 0.423 ± 0.011 (E15) |

| Paired gap (artery Dice, test) | Mean | 95 % CI (seeds) | Per seed | Validation |
|---|---|---|---|---|
| D − A | +0.000 | [−0.041, +0.040] | +0.019 / −0.012 / −0.007 | +0.007 |
| C − B | −0.001 | [−0.036, +0.034] | −0.013 / +0.015 / −0.005 | −0.000 |
| B − A | −0.016 | [−0.049, +0.017] | −0.002 / −0.029 / −0.018 | −0.008 |
| C − A | −0.017 | [−0.028, −0.006] | −0.015 / −0.014 / −0.022 | −0.009 |
| A + degradations − A | +0.010 | [−0.023, +0.042] | +0.023 / +0.008 / −0.003 | +0.008 (+0.011 / +0.008 / +0.004) |

| Artery Dice on degraded images | A | A* | D | A + degradations |
|---|---|---|---|---|
| Clean image | 0.428 | 0.424 | 0.427 | 0.437 |
| Poisson, levels 0 / 3 / 6 / 9 | 0.405 / 0.390 / 0.351 / 0.023 | 0.388 / 0.372 / 0.317 / 0.039 | 0.395 / 0.379 / 0.343 / 0.080 | 0.417 / 0.409 / 0.390 / 0.233 |
| JPEG, levels 6 / 9 | 0.415 / 0.032 | 0.417 / 0.073 | 0.420 / 0.137 | 0.432 / 0.251 |
| Blur, levels 3 / 6 / 9 | 0.390 / 0.314 / 0.206 | 0.394 / 0.321 / 0.210 | 0.393 / 0.288 / 0.181 | 0.423 / 0.398 / 0.348 |

**Conclusions**

- A and D reproduce E10 within ±0.005, C +0.007. B gains 0.019 (0.393 → 0.411, gap computed on the
  unrounded values); only the augmentation seed and the execution from a frozen commit changed, and
  the cause was not isolated.
- **D ≈ A confirmed** (+0.000): training on reconstructions brings nothing measurable.
- **Fine-tuning MedVAE on ARCADE has no effect on segmentation from the latent** (C − B = −0.001,
  signs −/+/−; same on the validation set). The wording "helps slightly" of E10 and of the report is
  withdrawn.
- B and C reach 96 % of the Dice of A (0.411 and 0.410 against 0.428), but the R control of E16 does
  almost as well.
- **A* ≈ A over 3 seeds** (−0.004, slightly below for all 3).
- **Robustness: the conclusions of E12 hold.** Under Poisson noise, A* does worse than A at levels 0
  to 6 (−0.017, −0.018, −0.034); D ≈ A up to level 6; at level 9, Poisson and JPEG make all three
  models collapse (Dice ≤ 0.14, D slightly less), and blur halves the Dice (0.18 to 0.21).
- **The degradation augmentation no longer costs anything on clean images**: +0.010 on the test set
  (2 seeds out of 3), +0.008 on the validation set (3 seeds out of 3), against −0.007 in E15. Gains
  under degradation: +0.21 (Poisson), +0.22 (JPEG) and +0.14 (blur) at level 9, +0.02 to +0.08 at
  level 6. Following the protocol (decision on the validation set), it could be turned on by
  default; this would require retraining B, C and D to keep the conditions comparable. It remains an
  option, off by default: official results = E17 without this augmentation.
- `downstream.py` paired the A and D runs in folder order (A seed 42 with D seed 44, etc.). The
  degraded images do not depend on this pairing: the means are correct; the D − A gaps above are
  paired by seed from the runs. Fixed (commit `04a8e11`).
- Results promoted: seed 42 of A, A*, B, C and D of E17.

## E18 — FiLM with scores A and B on 3 seeds (2026-09-29)

**Why**: the audit points out that E13 studies on 3 seeds the only unusable score (C, almost
constant in training) and that the A and B controls had a single seed. Scores A and B have similar
distributions in training (means 0.45 and 0.49) and on the test set (0.44 and 0.46).
**Protocol**: identical to E13 (`train_film.py`, 256×256, 3000 steps, test = seg_val), baseline,
FiLM-A and FiLM-B for seeds 42, 43 and 44, all at commit `4b6cf64` from the frozen worktree (runs
`2026-09-29_*_e18_film_{baseline,A,B}_seed4?`, all clean). Analysis `experiments/film/e18/` (CI over
images for the controls of a single model, over seeds for FiLM − baseline).

| Test PSNR (dB) | Baseline | FiLM-A | FiLM-B |
|---|---|---|---|
| True c | 42.258 ± 0.004 | 42.296 ± 0.020 | 42.307 ± 0.021 |
| Shuffled c | — | 42.289 | 42.274 |
| Constant c | — | 42.291 | 42.288 |

| Gap (PSNR, dB) | Mean | 95 % CI | Per seed |
|---|---|---|---|
| FiLM-A: true c − shuffled c | +0.007 | [+0.002, +0.014] (images) | +0.009 / +0.006 / +0.007 |
| FiLM-A: true c − constant c | +0.005 | [+0.002, +0.009] (images) | +0.006 / +0.005 / +0.005 |
| FiLM-B: true c − shuffled c | **+0.033** | [+0.018, +0.049] (images) | +0.037 / +0.031 / +0.032 |
| FiLM-B: true c − constant c | +0.020 | [+0.008, +0.031] (images) | +0.023 / +0.018 / +0.019 |
| FiLM-A − baseline | +0.038 | [−0.011, +0.087] (seeds) | +0.061 / +0.029 / +0.025 |
| FiLM-B − baseline | +0.050 | [−0.003, +0.102] (seeds) | +0.074 / +0.039 / +0.036 |

(on the vessels: FiLM − baseline +0.031 [+0.016, +0.047] with A, +0.038 [+0.018, +0.057] with B)

**Conclusions**

- **With scores A and B, the network uses the quality information**, reproducibly (true c >
  controls for all 3 seeds), but the effect is tiny: < 0.04 dB (max +0.033, score B).
- FiLM gains +0.04 to +0.05 dB over the baseline, of which at most 0.02-0.03 dB come from the score;
  the rest comes from the added parameters. The FiLM − baseline gap is not established on the whole
  image (CI over seeds containing 0); it is on the vessels.
- Conclusion of part B: conditioning MedVAE on a quality score computed from the image brings no
  useful gain. Score C is not used (it barely varies in training); A and B are, barely.
- The E18 baseline reproduces that of E13 (42.258 against 42.261 dB). As in E13, the best step is
  the last one (3000) for 8 runs out of 9: the conclusions hold for this budget.

## E19 — JEPA on 3 pre-trainings (2026-09-29)

**Why**: the conclusions of E14 rested on a single pre-training per stage, and the audit points out
two errors I made in writing them: the ranking of the probe was **not** the same in all three
settings (at 20 and 60 epochs, stage 2 matched or beat MedVAE: 0.393 against 0.391 at 20 epochs,
0.454 against 0.452 at 60), and the probe setting had been chosen by looking at the test set. Two
pipeline flaws came on top of that: the Gaussian augmentation noise of `jepa_adaptation` used the
albumentations 2 default (standard deviation 0.2-0.44 on [0, 1], much stronger than intended), and,
as for segmentation, the augmentations repeated across workers and epochs; finally, the stage-2
validation (which picks `best.pt`) drew different random masks at every call.
**Change** (commit `4b6cf64`): noise brought back to the intended standard deviation
(0.012-0.028), per-worker augmentation seed, validation masks drawn with a fixed generator,
`--seed` and `--run-name` options (traceable runs in `experiments/runs/`).
**Protocol**: stage 1 then stage 2 (15 epochs each, as in E14) for seeds 42, 43 and 44, from the
frozen worktree; then, for each model, vessel / background linear probe (100 epochs, lr 1e-3 and
1e-2, 3 probe seeds), reconstruction and effective rank on the 300 test images. **The probe lr is
chosen on the validation set** (seg_val): 1e-2 for all three models. MedVAE, pre-trained and fixed,
has a single model (3 probe seeds).
**Runs**: `experiments/runs/2026-09-29_*_e19_jepa_stage{1,2}_seed4?` (all clean); results
`experiments/jepa/e19/`.

| Test (mean ± standard deviation over 3 pre-trainings) | MedVAE (×4, pre-trained) | Stage 1 (×8, from scratch) | Stage 2 JEPA |
|---|---|---|---|
| Vessel linear probe, Dice (lr 1e-2) | 0.517 ± 0.005 (probe seeds) | **0.561 ± 0.006** | 0.536 ± 0.011 |
| Vessel IoU | 0.349 | **0.390** | 0.366 |
| Validation Dice of the probe | 0.570 | **0.604** | 0.569 |
| Reconstruction: PSNR / vessels (dB) | 34.8 / 35.5 | 31.1 / 31.7 | **18.6** / 22.9 |
| Effective rank of the features (channels) | 251 (512) | 147 (152 / 153 / 136) | 156 (157 / 150 / 161) |
| Components for 90 % of the variance | 61 | 46 (51 / 53 / 35) | 50 (50 / 44 / 55) |

| Paired gap per pre-training (probe Dice, test) | Mean | 95 % CI (seeds) | Per seed |
|---|---|---|---|
| Stage 2 − stage 1 | **−0.025** | [−0.045, −0.005] | −0.019 / −0.022 / −0.034 |
| Stage 1 − MedVAE | +0.043 | [+0.029, +0.058] | +0.039 / +0.050 / +0.041 |
| Stage 2 − MedVAE | +0.019 | [−0.008, +0.045] | +0.021 / +0.028 / +0.007 |

(with lr 1e-3: stage 2 − stage 1 = −0.030, 3 seeds out of 3; same direction on the validation set:
−0.035 [−0.056, −0.014])

**Conclusions**

- **JEPA lowers the quality of the representation it sets out to improve**: −0.025 probe Dice
  relative to the stage 1 it starts from, for all 3 pre-trainings, on the test set as on the
  validation set. The direction of E14 is confirmed, the size is half as large (−0.056 in E14).
- **Stage 2 is not below MedVAE** (+0.019, CI containing 0): the E14 sentence "below the original
  MedVAE" is withdrawn.
- Stage 1 beats MedVAE (+0.043), but the two models differ in architecture (256 against 512
  channels), compression rate (×8 against ×4 per side) and data (ARCADE against radiographs and
  mammograms): this gap says nothing about JEPA and is not attributed to any one of these factors.
- **Effective rank: no net effect of JEPA** (+9, CI [−27, +45], 2 seeds out of 3 up). The
  111 → 139 increase of E14 does not reproduce; what holds is the absence of dimensional
  contraction.
- The reconstruction collapses at stage 2 (31.1 → 18.6 dB) for all 3 seeds: the frozen decoder does
  not follow the encoder modified by JEPA.
- The levels changed since E14 (stage 1: 0.541 → 0.561, rank 111 → 147; stage 2: 0.485 → 0.536).
  In between, the augmentation noise, the worker seed, the stage-2 validation and the seeds changed
  at the same time: these gaps are not attributed to any one of these causes. E19 replaces E14 for
  the figures of part C.
- **Limitation**: both stages, trained for 15 epochs as in the original study, did not converge
  (best validation loss at epoch 14 or 15 for each pre-training, stage 1 as stage 2; logs
  `jepa_adaptation/jobs/logs/e19_s4?_*.out`). A longer training could change the levels (audit
  MIN-16).

## E20 — Vessel visibility: contrast-to-noise ratio (2026-09-30)

**Question**: E11 shows that a simple 5×5 filter denoises better than MedVAE in PSNR. But a filter
blurs thin vessels: on vessel **visibility**, a standard measure in angiography, does the ranking
hold?
**Measurement** (`medvae_eval/robustness/cnr.py`): CNR = (mean of the local background − mean on
the vessel centreline) / standard deviation of the local background. Geometry taken from the
annotations of the clean image: centreline = skeleton of the mask, local width = 2 × distance to the
edge; **thin** vessels = annotated width ≤ 6 px (lower quartile), **thick** = ≥ 12 px (upper
quartile); local background = ring of 3 to 13 px around the mask, outside the collimator, attached
to the nearest centreline point. Same 100 seg_val images, same degradations (10 levels) and same
seeded noise as E11; versions compared: degraded input, MedVAE, 5×5 Gaussian and median filters.
95 % bootstrap CIs over images (same images for every version).
**Results**: `experiments/robustness/e20_cnr/` (`summary.md`, `cnr.png`, `mechanism.csv`)

| CNR (100 images) | Thin: degraded | MedVAE | Gaussian 5×5 | median 5×5 | Thick: degraded | MedVAE | Gaussian 5×5 | median 5×5 |
|---|---|---|---|---|---|---|---|---|
| Clean image (ref.) | 0.88 | 0.85 | — | — | 1.91 | 1.90 | — | — |
| Poisson, level 0 | 0.72 | 0.73 | **0.81** | 0.76 | 1.54 | 1.70 | 1.97 | **2.00** |
| Poisson, level 6 | 0.58 | 0.64 | **0.77** | 0.72 | 1.25 | 1.53 | 1.90 | **1.92** |
| Poisson, level 9 | 0.30 | 0.45 | **0.62** | 0.55 | 0.62 | 1.16 | **1.49** | 1.48 |
| JPEG, level 6 | **0.88** | 0.86 | 0.83 | 0.79 | 1.94 | 1.90 | 2.01 | **2.02** |
| Blur, level 6 | 0.50 | 0.50 | 0.49 | 0.48 | 1.92 | 1.92 | 1.90 | 1.89 |

| Paired gap (Poisson, levels 0 / 3 / 6 / 9) | Thin | Thick |
|---|---|---|
| MedVAE − degraded | +0.01 / +0.01 / +0.06 / +0.15 (CI containing 0 at levels 0 and 3) | +0.16 / +0.20 / +0.29 / +0.54 |
| MedVAE − Gaussian 5×5 | −0.08 / −0.11 / −0.13 / −0.16 (CI excluding 0) | −0.27 / −0.31 / −0.36 / −0.33 |

Mechanism (Poisson, levels 0 → 9, `mechanism.csv`): on thick vessels, MedVAE and the Gaussian
filter keep all the contrast (≥ 0.96 of the clean contrast), but the filter removes more noise
(background standard deviation 0.079 → 0.092 against 0.088 → 0.120 for MedVAE). On thin vessels,
the filter keeps 85 to 90 % of the contrast; MedVAE 94 % at level 0 but 68 % at level 9, and it
removes less noise.

**Conclusions**

- **The CNR confirms the PSNR ranking, thin vessels included**: under Poisson noise, MedVAE makes
  the vessels more visible than the noisy input (clearly for thick ones, barely for thin ones at low
  noise), but a simple 5×5 filter does better at every level, for thin and thick vessels alike. The
  intuition "the filter erases thin vessels, MedVAE does not" does not hold at this scale (thin
  vessels of 6 px or less).
- **With no noise to remove, MedVAE changes almost nothing**: on the clean image, JPEG and blur, the
  CNR follows the input (−0.03 on the thin vessels of the clean image), whereas the filters, which
  only blur, lower that of thin vessels (median: −0.06 to −0.07 relative to MedVAE). Consistent with
  A* ≈ A (E16, E17): as a preprocessor of clean images, MedVAE is harmless.
- **Limits of the CNR**: the background standard deviation on the clean image (≈ 0.08, i.e. ~20
  grey levels) mostly comes from the anatomical texture, not from quantum noise; thin vessels
  therefore have a CNR below 1 even on the clean image. The CNR rewards smoothing (the Gaussian
  filter exceeds the clean image on thick vessels, 2.01 against 1.91) and ignores the texture of the
  noise. The rigorous measure would be the detectability index d′ of a model observer (channelised
  Hotelling), discussed in the report.
- The script was run from the uncommitted working tree (hash in `provenance.txt`), during the night
  of 29 to 30 September, then committed as is with the results (identical sha256).
