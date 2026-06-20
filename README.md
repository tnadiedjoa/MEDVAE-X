# MedVAE: Adaptation, Fine-tuning and Analysis for Medical Imaging and Coronary Segmentation


## JEPA adaptation of phase 2 medvae training (ROTHLINGSHOFER Yanic)

This project explores an adaptation of the second training stage of Med-VAE. In
the original pipeline, stage 2 relies on BioMedCLIP to preserve clinically
relevant information in the latent space. The goal here is to replace this
external vision-language supervision with a self-supervised JEPA objective,
directly learned from medical images.

The motivation is to make the adaptation less dependent on BioMedCLIP, whose
representations may not always match the target medical domain or imaging
modality. JEPA is a good fit because it trains the model to predict missing
latent information from visible context, encouraging semantic and structural
representations without reconstructing pixels. The design is inspired by I-JEPA:
a context encoder processes visible image patches, a frozen target encoder
provides target latent representations, and a predictor is trained with a latent
prediction loss.

![JEPA adaptation pipeline](jepa_adaptation/pipeline_figure/pipeline.png)
*Figure: JEPA adaptation pipeline, inspired by the original Med-VAE pipeline figure.*

In practice, the Med-VAE encoder is reused as the backbone of the JEPA adapted
encoder. The BioMedCLIP-based consistency term is replaced by a latent
prediction loss between predicted target latents and frozen target encoder
latents. The adapted encoder can then be evaluated on downstream medical image
tasks.

### 2026-05-26

- Integrated MedMNIST dataset for pretraining (which contains 8 2d images datasets ~ 450k images).
- Completed the full pretraining pipeline: phase 1 (VAE reconstruction) and phase 2 (JEPA latent prediction) trainers are implemented in `utils/vae_trainer.py` and `utils/jepa_trainer.py`, driven by a unified entry-point `pretraining.py`.
- Added a downstream evaluation pipeline (`downstream.py`, `utils/downstream_wrapper.py`) supporting linear probing on top of the frozen JEPA-adapted encoder.
- Added SLURM job scripts (`jobs/`) for cluster execution of all training phases and downstream evaluation.
- Configuration files reorganised: `configs/pretraining.yaml` for the pretraining phases, `configs/downstream.yaml` for evaluation.
- General code cleanup across model modules (removed dead code, fixed imports, unified logging).

## Fine-tuning MedVAE on New Medical Modalities (PALAGI Théo)

This project investigates whether fine-tuning MedVAE, which is a medical image autoencoder pre-trained on chest X-rays and mammographies, on a new imaging modality can improve downstream segmentation performance.  

We use the ARCADE dataset, which contains coronary angiography images annotated with 26 arterial segmentation classes. Coronary angiographies are structurally very different from the modalities MedVAE was trained on: they feature thin tubular structures, bifurcations, and stenosis regions that require fine-grained spatial encoding to be preserved under compression.
The core hypothesis is that a general-purpose medical encoder, while useful, may not capture the domain-specific visual features needed for precise vascular segmentation. Fine-tuning MedVAE on ARCADE images should push its latent space to better represent these structures, leading to better downstream performance.  

To test this, we design three comparable pipelines. The first trains a standard U-Net directly on full-resolution ARCADE images and serves as an upper-bound reference. The second uses the pre-trained MedVAE encoder to compress images into latent representations, which are then passed to a lightweight segmentation head, this measures how well the general model transfers to this new modality. The third repeats the second pipeline but with a MedVAE encoder that has been fine-tuned on ARCADE images beforehand, isolating the contribution of domain adaptation.
All three pipelines are evaluated using the mean Dice score across the 26 arterial classes on a held-out test set. The gap between the second and third conditions directly quantifies the benefit of fine-tuning MedVAE on a previously unseen medical modality.

![fine-tune pipeline](finetune/images/pipeline.jpg)
*Figure: Experimental pipelines for evaluating MedVAE adaptation on coronary angiography segmentation.*

## Input quality analysis for MedVAE reconstruction (CORLOU Elias & NADIEDJOA Théophile)

This work evaluates how MedVAE's reconstruction fidelity behaves when the input image is degraded rather than clean, a question directly motivated by low-dose acquisition noise, compression, and motion artefacts in coronary angiography. An initial exploration using no-reference quality metrics (ARNIQA, then a custom engineered score) was discarded: on clean images the intrinsic quality variance is too small to be exploitable, and under controlled degradation these scores are dominated by blur, which makes them non-monotone and unsuitable as a clean quality axis.

We instead adopt a fully full-reference protocol: since the clean image, its degraded version, and MedVAE's reconstruction are all available simultaneously, reconstruction fidelity can be measured by direct comparison rather than through a no-reference proxy. Using a vessel-masked PSNR (mPSNR, restricted to ARCADE's coronary annotations), we run independent sweeps over ~50 levels for three degradation types — Poisson noise, Gaussian blur, and JPEG compression — and correlate mPSNR(degraded, reconstruction) against the degradation amplitude mPSNR(clean, degraded).

The results reveal that MedVAE's behaviour depends on the spectral nature of the degradation, not just its intensity. Gaussian blur shows a strong anti-correlation (r ≈ −0.998): blurred inputs reconstruct better, consistent with the model's spectral bias towards low frequencies. Poisson noise shows a strong positive correlation (r ≈ +0.998): MedVAE's latent bottleneck discards stochastic high-frequency noise, acting as a genuine but only partial denoiser. JPEG compression is non-monotone, combining a "blur-like" regime that initially helps reconstruction with a low-quality regime where structured block artefacts collapse it; global correlation is misleading (r ≈ +0.03) but each regime taken separately is nearly perfectly correlated. Overall, MedVAE acts as an intelligent low-pass filter and non-linear denoiser, but remains vulnerable to structured out-of-distribution high frequencies it can neither encode nor ignore.

![quality pipeline](medvae_eval/pipeline_figure/pipeline_3.jpg)

*Figure: Controlled full-reference degradation pipeline for the MedVAE robustness analysis.*
