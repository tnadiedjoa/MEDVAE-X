import torch
import torch.nn as nn

from .modules.context_encoder import ContextEncoder, sample_patch_mask
from .modules.jepa_losses import LatentPredictionLoss
from .modules.jepa_predictor import JEPAPredictor
from .modules.target_encoder import TargetEncoder


class MedVAE_JEPA(nn.Module):
    """JEPA adaptation wrapper for Med-VAE stage 2."""

    def __init__(
        self,
        autoencoder,
        latent_dim,
        patch_size=16,
        target_mask_ratio=0.4,
        predictor_hidden_dim=None,
        predictor_depth=3,
        spatial_dims=2,
        ema_momentum=0.996,
        loss_type="smooth_l1",
        normalize_loss=True,
        sample_posterior=False,
        freeze_decoder=True,
    ):
        super().__init__()
        self.context_encoder = ContextEncoder(
            autoencoder=autoencoder,
            patch_size=patch_size,
            sample_posterior=sample_posterior,
            freeze_decoder=freeze_decoder,
        )
        self.target_encoder = TargetEncoder(self.context_encoder)
        self.predictor = JEPAPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            depth=predictor_depth,
            spatial_dims=spatial_dims,
            use_target_mask=True,
        )
        self.criterion = LatentPredictionLoss(
            loss_type=loss_type,
            normalize=normalize_loss,
        )
        self.patch_size = patch_size
        self.target_mask_ratio = target_mask_ratio
        self.ema_momentum = ema_momentum
        self.spatial_dims = spatial_dims

    def sample_masks(self, x, generator=None):
        target_mask = sample_patch_mask(
            batch_size=x.shape[0],
            spatial_shape=x.shape[2:],
            patch_size=self.patch_size,
            mask_ratio=self.target_mask_ratio,
            device=x.device,
            generator=generator,
        )
        context_mask = ~target_mask
        return context_mask, target_mask

    def forward(
        self,
        x,
        context_mask=None,
        target_mask=None,
        generator=None,
        return_loss=True,
    ):
        if context_mask is None or target_mask is None:
            sampled_context, sampled_target = self.sample_masks(x, generator=generator)
            context_mask = sampled_context if context_mask is None else context_mask
            target_mask = sampled_target if target_mask is None else target_mask

        context_outputs = self.context_encoder(x, keep_mask=context_mask)
        target_outputs = self.target_encoder(x, keep_mask=target_mask)
        predicted_target = self.predictor(
            context_outputs["latent"], target_mask=target_mask
        )

        outputs = {
            "context_latent": context_outputs["latent"],
            "target_latent": target_outputs["latent"],
            "predicted_target_latent": predicted_target,
            "context_mask": context_mask,
            "target_mask": target_mask,
            "context_input": context_outputs["input"],
            "target_input": target_outputs["input"],
        }

        if return_loss:
            loss, logs = self.criterion(
                predicted_target,
                target_outputs["latent"],
                target_mask=target_mask,
            )
            outputs["loss"] = loss
            outputs["logs"] = logs

        return outputs

    @torch.no_grad()
    def update_target_encoder(self, momentum=None):
        self.target_encoder.update_momentum(
            self.context_encoder,
            momentum=self.ema_momentum if momentum is None else momentum,
        )

    def trainable_parameters(self):
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())

    def adapted_encoder(self):
        return self.context_encoder.autoencoder.encoder
