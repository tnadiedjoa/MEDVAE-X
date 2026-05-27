# projet_IM06


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

![JEPA adaptation pipeline](jepa-adaptation/pipeline_figure/pipeline.png)
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

This work investigates whether the quality of the input medical images impacts MedVAE reconstruction performance. As a first exploratory step, we evaluated the clean ARCADE dataset using the no-reference metric ARNIQA, alongside a composite image quality score based on several classical sharpness and perceptual metrics. Reconstruction quality was then measured with PSNR and MS-SSIM after MedVAE inference. Initial correlation analyses did not reveal clear or reliable trends, likely due to the domain mismatch of ARNIQA, which is trained on natural images rather than medical data. The next step is therefore to apply controlled synthetic degradations (noise, blur, compression) to the dataset in order to study how reconstruction performance evolves with progressively degraded inputs. Additional implementation details and analyses are available in `medvae_eval/metrics_recap.md`

![Input quality analysis pipeline](medvae_eval/pipeline_figure/pipeline_2.jpg)

*Figure: Input quality analysis pipeline, inspired by the original Med-VAE pipeline figure.*
