import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mvae_jepa import _resize_mask

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI_3 = math.sqrt(2.0 * math.pi / 3.0)


def epps_pulley(projections):
    diff = projections.unsqueeze(0) - projections.unsqueeze(1)
    term1 = torch.exp(-0.5 * diff.pow(2)).mean(dim=(0, 1))
    term2 = torch.exp(-0.25 * projections.pow(2)).mean(dim=0)
    stat = _SQRT_2PI * term1 - 2.0 * _SQRT_PI * term2 + _SQRT_2PI_3
    return stat.mean()


class SIGRegLoss(nn.Module):
    def __init__(self, num_projections=64, max_tokens=512):
        super().__init__()
        self.num_projections = num_projections
        self.max_tokens = max_tokens

    def _embeddings(self, latent):
        emb = latent.permute(0, *range(2, latent.dim()), 1).reshape(-1, latent.shape[1])
        if self.max_tokens is not None and emb.shape[0] > self.max_tokens:
            idx = torch.randperm(emb.shape[0], device=emb.device)[: self.max_tokens]
            emb = emb[idx]
        return emb

    def forward(self, latent):
        emb = self._embeddings(latent)
        directions = torch.randn(
            emb.shape[1], self.num_projections, device=emb.device, dtype=emb.dtype
        )
        directions = F.normalize(directions, dim=0)
        projections = emb @ directions
        return epps_pulley(projections)


class Stage2Loss(nn.Module):
    def __init__(
        self,
        loss_type="smooth_l1",
        normalize=True,
        sigreg_weight=1.0,
        num_projections=64,
        max_tokens=512,
    ):
        super().__init__()
        if loss_type not in {"mse", "smooth_l1"}:
            raise ValueError("loss_type must be 'mse' or 'smooth_l1'.")
        self.loss_type = loss_type
        self.normalize = normalize
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGRegLoss(num_projections=num_projections, max_tokens=max_tokens)

    def prediction_loss(self, predicted_latent, target_latent, target_mask=None):
        target_latent = target_latent.detach()

        if self.normalize:
            dims = (0,) + tuple(range(2, target_latent.dim()))
            mean = target_latent.mean(dim=dims, keepdim=True)
            std = target_latent.std(dim=dims, keepdim=True).clamp_min(1e-5)
            target_latent = (target_latent - mean) / std
            predicted_latent = (predicted_latent - mean) / std

        if self.loss_type == "mse":
            loss = (predicted_latent - target_latent).pow(2)
        else:
            loss = F.smooth_l1_loss(predicted_latent, target_latent, reduction="none")

        loss = loss.mean(dim=1, keepdim=True)
        mask = _resize_mask(target_mask, predicted_latent)
        if mask is None:
            mask = torch.ones_like(loss)
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom

    def forward(self, outputs, split="train"):
        pred_loss = self.prediction_loss(
            outputs["predicted_target_latent"],
            outputs["target_latent"],
            target_mask=outputs["target_mask"],
        )
        if self.sigreg_weight > 0:
            sigreg_loss = self.sigreg(outputs["context_latent"])
        else:
            sigreg_loss = torch.zeros((), device=pred_loss.device)
        loss = pred_loss + self.sigreg_weight * sigreg_loss

        log = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/prediction_loss": pred_loss.detach(),
            f"{split}/sigreg_loss": sigreg_loss.detach(),
        }
        return loss, log
