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

### Progress summary — 2026-05-26

- Completed the full pretraining pipeline: phase 1 (VAE reconstruction) and phase 2 (JEPA latent prediction) trainers are implemented in `utils/vae_trainer.py` and `utils/jepa_trainer.py`, driven by a unified entry-point `pretraining.py`.
- Added a downstream evaluation pipeline (`downstream.py`, `utils/downstream_wrapper.py`) supporting linear probing on top of the frozen JEPA-adapted encoder.
- Added SLURM job scripts (`jobs/`) for cluster execution of all training phases and downstream evaluation.
- Configuration files reorganised: `configs/pretraining.yaml` for the pretraining phases, `configs/downstream.yaml` for evaluation.
- General code cleanup across model modules (removed dead code, fixed imports, unified logging).
