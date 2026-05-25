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
