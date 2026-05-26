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
    """Return a boolean patch mask with True entries selected as targets."""
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
    """Upsample a patch mask to image resolution with nearest-neighbor blocks."""
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


class ContextEncoder(nn.Module):
    """Med-VAE encoder wrapper used on visible image patches."""

    def __init__(
        self,
        autoencoder,
        patch_size=16,
        sample_posterior=False,
        freeze_decoder=True,
    ):
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


def resize_patch_mask_to_latent(mask, latent):
    if mask is None:
        return None

    mode = "nearest"
    resized = F.interpolate(mask.float(), size=latent.shape[2:], mode=mode)
    return resized.to(dtype=torch.bool, device=latent.device)
