import copy

import torch
import torch.nn as nn

from .context_encoder import apply_patch_keep_mask


class TargetEncoder(nn.Module):
    def __init__(self, context_encoder):
        super().__init__()
        self.encoder = copy.deepcopy(context_encoder)
        self.freeze()

    def freeze(self):
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_momentum(self, context_encoder, momentum=0.996):
        target_params = self.encoder.parameters()
        context_params = context_encoder.parameters()
        for target_param, context_param in zip(target_params, context_params):
            target_param.data.mul_(momentum).add_(
                context_param.data, alpha=1.0 - momentum
            )

        target_buffers = self.encoder.buffers()
        context_buffers = context_encoder.buffers()
        for target_buffer, context_buffer in zip(target_buffers, context_buffers):
            target_buffer.copy_(context_buffer)

    @torch.no_grad()
    def forward(self, x, keep_mask=None):
        masked_x = apply_patch_keep_mask(x, keep_mask, self.encoder.patch_size)
        latent, posterior, z = self.encoder.encode(masked_x)
        return {
            "latent": latent.detach(),
            "posterior": posterior,
            "z": z.detach(),
            "input": masked_x,
            "keep_mask": keep_mask,
        }
