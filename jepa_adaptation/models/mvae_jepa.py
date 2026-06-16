import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple(value, spatial_dims):
    if isinstance(value, int):
        return (value,) * spatial_dims
    if len(value) != spatial_dims:
        raise ValueError(f"Expected {spatial_dims} values, got {value}.")
    return tuple(value)


def sample_patch_mask(
    batch_size,
    spatial_shape,
    patch_size,
    mask_ratio,
    device=None,
    generator=None,
):
    spatial_dims = len(spatial_shape)
    patch_size = _as_tuple(patch_size, spatial_dims)
    grid_shape = tuple((s + p - 1) // p for s, p in zip(spatial_shape, patch_size))
    num_patches = 1
    for size in grid_shape:
        num_patches *= size

    num_masked = max(1, min(num_patches, int(round(num_patches * mask_ratio))))
    scores = torch.rand(batch_size, num_patches, device=device, generator=generator)
    selected = scores.argsort(dim=1)[:, :num_masked]
    flat_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    flat_mask.scatter_(1, selected, True)
    return flat_mask.view(batch_size, 1, *grid_shape)


def patch_mask_to_image_mask(patch_mask, image_shape, patch_size):
    spatial_dims = len(image_shape)
    patch_size = _as_tuple(patch_size, spatial_dims)
    image_mask = patch_mask
    for dim, repeat in enumerate(patch_size, start=2):
        image_mask = image_mask.repeat_interleave(repeat, dim=dim)

    slices = (slice(None), slice(None)) + tuple(slice(0, s) for s in image_shape)
    return image_mask[slices]


def apply_patch_keep_mask(x, keep_mask, patch_size, mask_value=0.0):
    if keep_mask is None:
        return x

    image_mask = patch_mask_to_image_mask(keep_mask, x.shape[2:], patch_size)
    image_mask = image_mask.to(device=x.device, dtype=x.dtype)
    return x * image_mask + mask_value * (1.0 - image_mask)


def _resize_mask(mask, latent):
    if mask is None:
        return None
    resized = F.interpolate(mask.float(), size=latent.shape[2:], mode="nearest")
    return resized.to(device=latent.device, dtype=latent.dtype)


class ContextEncoder(nn.Module):
    def __init__(self, autoencoder, patch_size=16, sample_posterior=False, freeze_decoder=True):
        super().__init__()
        self.autoencoder = autoencoder
        self.patch_size = patch_size
        self.sample_posterior = sample_posterior

        if freeze_decoder and hasattr(self.autoencoder, "decoder"):
            self.autoencoder.decoder.requires_grad_(False)
        if freeze_decoder and hasattr(self.autoencoder, "post_quant_conv"):
            self.autoencoder.post_quant_conv.requires_grad_(False)

    def encode(self, x):
        z, posterior, latent = self.autoencoder.compute_latent_proj(
            x, sample_posterior=self.sample_posterior
        )
        if latent is None:
            latent = z
        return latent, posterior, z

    def forward(self, x, keep_mask=None):
        masked_x = apply_patch_keep_mask(x, keep_mask, self.patch_size)
        latent, posterior, z = self.encode(masked_x)
        return {
            "latent": latent,
            "posterior": posterior,
            "z": z,
            "input": masked_x,
            "keep_mask": keep_mask,
        }


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
        for target_param, context_param in zip(
            self.encoder.parameters(), context_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(context_param.data, alpha=1.0 - momentum)

        for target_buffer, context_buffer in zip(
            self.encoder.buffers(), context_encoder.buffers()
        ):
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


class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=None, depth=3, spatial_dims=2, use_target_mask=True):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1.")

        hidden_dim = hidden_dim or latent_dim * 2
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        norm = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
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

    def forward(self, context_latent, target_mask=None):
        predictor_input = context_latent
        if self.use_target_mask:
            mask = _resize_mask(target_mask, context_latent)
            if mask is None:
                mask = torch.zeros_like(context_latent[:, :1])
            predictor_input = torch.cat([context_latent, mask], dim=1)

        return self.net(predictor_input)


class MVAE_JEPA(nn.Module):
    def __init__(
        self,
        mvae,
        latent_dim=None,
        patch_size=16,
        target_mask_ratio=0.4,
        predictor_hidden_dim=None,
        predictor_depth=3,
        ema_momentum=0.996,
        sample_posterior=False,
        freeze_decoder=True,
    ):
        super().__init__()
        autoencoder = mvae.model
        spatial_dims = 3 if getattr(mvae, "is_3d", False) else 2
        latent_dim = latent_dim or autoencoder.embed_dim

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

    def forward(self, x, context_mask=None, target_mask=None, generator=None):
        if context_mask is None or target_mask is None:
            sampled_context, sampled_target = self.sample_masks(x, generator=generator)
            context_mask = sampled_context if context_mask is None else context_mask
            target_mask = sampled_target if target_mask is None else target_mask

        context_outputs = self.context_encoder(x, keep_mask=context_mask)
        target_outputs = self.target_encoder(x, keep_mask=target_mask)
        predicted_target = self.predictor(
            context_outputs["latent"], target_mask=target_mask
        )

        return {
            "context_latent": context_outputs["latent"],
            "target_latent": target_outputs["latent"],
            "predicted_target_latent": predicted_target,
            "context_mask": context_mask,
            "target_mask": target_mask,
            "context_input": context_outputs["input"],
            "target_input": target_outputs["input"],
        }

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
