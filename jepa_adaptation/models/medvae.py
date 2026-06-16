import torch
import torch.nn as nn

try:
    from .modules.autoencoder_kl import AutoencoderKL
    from .modules.autoencoder_kl_3d import AutoencoderKL as AutoencoderKL_3D
except ImportError:
    from modules.autoencoder_kl import AutoencoderKL
    from modules.autoencoder_kl_3d import AutoencoderKL as AutoencoderKL_3D


_MEDVAE_HF_REPO = "stanfordmimi/MedVAE"
_MEDVAE_CKPTS = {
    "medvae_4_1_2d": "model_weights/vae_4x_1c_2D.ckpt",
    "medvae_4_3_2d": "model_weights/vae_4x_3c_2D.ckpt",
    "medvae_4_4_2d": "model_weights/vae_4x_4c_2D.ckpt",
    "medvae_8_1_2d": "model_weights/vae_8x_1c_2D.ckpt",
    "medvae_8_4_2d": "model_weights/vae_8x_4c_2D.ckpt",
    "medvae_4_1_3d": "model_weights/vae_4x_1c_3D.ckpt",
    "medvae_8_1_3d": "model_weights/vae_8x_1c_3D.ckpt",
}


def _download_medvae_ckpt(name):
    from huggingface_hub import hf_hub_download

    if name not in _MEDVAE_CKPTS:
        raise ValueError(f"Unknown MedVAE model: {name!r}")
    return hf_hub_download(repo_id=_MEDVAE_HF_REPO, filename=_MEDVAE_CKPTS[name])


class MVAE(nn.Module):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        spatial_dims=2,
        apply_channel_ds=True,
        medvae_pretrained=None,
        ckpt_path=None,
    ):
        super().__init__()
        self.is_3d = spatial_dims == 3
        self.embed_dim = embed_dim

        AE = AutoencoderKL_3D if self.is_3d else AutoencoderKL
        self.model = AE(
            ddconfig=ddconfig,
            embed_dim=embed_dim,
            apply_channel_ds=apply_channel_ds,
        )

        if medvae_pretrained is not None:
            self.load_weights(_download_medvae_ckpt(medvae_pretrained))
        elif ckpt_path is not None:
            self.load_weights(ckpt_path)

    def load_weights(self, path):
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, dict):
            sd = raw.get("state_dict", raw.get("autoencoder", raw))
        else:
            sd = raw
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(
            f"[MVAE] loaded {path} "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
        if missing:
            print(f"[MVAE] missing keys:")
            for k in missing:
                print(f"    {k}")
        if unexpected:
            print(f"[MVAE] unexpected keys:")
            for k in unexpected:
                print(f"    {k}")

    def encode(self, x):
        return self.model.encode(x).sample()

    def decode(self, z):
        return self.model.decode(z)

    def forward(self, x, decode=False):
        dec, _, latent = self.model(x, decode=decode)
        if decode:
            return dec, latent
        return latent
