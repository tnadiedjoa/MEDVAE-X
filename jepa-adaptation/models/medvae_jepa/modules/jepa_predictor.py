import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(spatial_dims):
    if spatial_dims == 2:
        return nn.Conv2d
    if spatial_dims == 3:
        return nn.Conv3d
    raise ValueError("spatial_dims must be 2 or 3.")


def _norm(spatial_dims):
    if spatial_dims == 2:
        return nn.BatchNorm2d
    if spatial_dims == 3:
        return nn.BatchNorm3d
    raise ValueError("spatial_dims must be 2 or 3.")


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim=None,
        depth=3,
        spatial_dims=2,
        use_target_mask=True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1.")

        hidden_dim = hidden_dim or latent_dim * 2
        conv = _conv(spatial_dims)
        norm = _norm(spatial_dims)
        in_channels = latent_dim + int(use_target_mask)

        layers = [
            conv(in_channels, hidden_dim, kernel_size=1),
            norm(hidden_dim),
            nn.SiLU(inplace=True),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    conv(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    norm(hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.append(conv(hidden_dim, latent_dim, kernel_size=1))

        self.net = nn.Sequential(*layers)
        self.use_target_mask = use_target_mask

    def _resize_mask(self, target_mask, latent):
        if target_mask is None:
            return None
        return F.interpolate(target_mask.float(), size=latent.shape[2:], mode="nearest")

    def forward(self, context_latent, target_mask=None):
        predictor_input = context_latent
        if self.use_target_mask:
            mask = self._resize_mask(target_mask, context_latent)
            if mask is None:
                mask = torch.zeros_like(context_latent[:, :1])
            predictor_input = torch.cat([context_latent, mask], dim=1)

        return self.net(predictor_input)
