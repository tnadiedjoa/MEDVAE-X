import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictionLoss(nn.Module):
    def __init__(self, loss_type="smooth_l1", normalize=True):
        super().__init__()
        if loss_type not in {"mse", "smooth_l1"}:
            raise ValueError("loss_type must be 'mse' or 'smooth_l1'.")
        self.loss_type = loss_type
        self.normalize = normalize

    def _resize_mask(self, target_mask, latent):
        if target_mask is None:
            return torch.ones_like(latent[:, :1], dtype=latent.dtype)
        mask = F.interpolate(target_mask.float(), size=latent.shape[2:], mode="nearest")
        return mask.to(device=latent.device, dtype=latent.dtype)

    def forward(self, predicted_latent, target_latent, target_mask=None):
        target_latent = target_latent.detach()

        if self.normalize:
            predicted_latent = F.normalize(predicted_latent, dim=1)
            target_latent = F.normalize(target_latent, dim=1)

        if self.loss_type == "mse":
            loss = (predicted_latent - target_latent).pow(2)
        else:
            loss = F.smooth_l1_loss(
                predicted_latent, target_latent, reduction="none"
            )

        loss = loss.mean(dim=1, keepdim=True)
        mask = self._resize_mask(target_mask, predicted_latent)
        denom = mask.sum().clamp_min(1.0)
        total = (loss * mask).sum() / denom

        with torch.no_grad():
            unmasked = loss.mean()
            masked_fraction = mask.mean()

        return total, {
            "jepa/latent_prediction_loss": total.detach(),
            "jepa/unmasked_latent_prediction_loss": unmasked.detach(),
            "jepa/target_mask_fraction": masked_fraction.detach(),
        }
