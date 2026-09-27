"""MedVAE conditionné par un score de qualité c via FiLM (repris du notebook 04 de l'axe B).

Après chaque ResnetBlock de l'encodeur et du décodeur : h ← (1 + γ_i(c)) · h + β_i(c),
avec (γ_i, β_i) produits par un MLP partagé et des têtes linéaires initialisées à zéro
(le modèle conditionné est identique au MedVAE de départ avant entraînement).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cvae"))
from medvae_standalone import AutoencoderKL, DiagonalGaussianDistribution  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_CKPT = REPO_ROOT / "medvae_eval" / "cvae" / "outputs_cvae" / "pretrained_weights" / "vae_4x_4c_2D.ckpt"

# Configuration de medvae_4_4_2d (vae_4x_4c_2D) : facteur 4 par côté, 4 canaux latents
DDCONFIG = {"double_z": True, "z_channels": 4, "resolution": 64, "in_channels": 1, "out_ch": 1,
            "ch": 128, "ch_mult": [1, 2, 4], "num_res_blocks": 2, "attn_resolutions": [],
            "dropout": 0.0}
EMBED_DIM = 4


def pretrained_ckpt() -> Path:
    """Poids officiels MedVAE vae_4x_4c_2D (téléchargés depuis HuggingFace si absents)."""
    if not PRETRAINED_CKPT.exists():
        import shutil
        from huggingface_hub import hf_hub_download
        PRETRAINED_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download("stanfordmimi/MedVAE", "model_weights/vae_4x_4c_2D.ckpt"), PRETRAINED_CKPT)
    return PRETRAINED_CKPT


def load_medvae() -> AutoencoderKL:
    return AutoencoderKL(ddconfig=DDCONFIG, embed_dim=EMBED_DIM, ckpt_path=str(pretrained_ckpt()),
                         apply_channel_ds=False)   # channel_ds : non utilisé par encode/decode


class FiLMConditioner(nn.Module):
    """c ∈ [0,1] → liste de (γ_i, β_i) de forme (B, ch_i, 1, 1), têtes initialisées à zéro."""

    def __init__(self, channels_per_block: list, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.heads = nn.ModuleList()
        for ch in channels_per_block:
            head = nn.Linear(hidden_dim, 2 * ch)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.heads.append(head)

    def forward(self, c: torch.Tensor) -> list:
        emb = self.embed(c.view(-1, 1))
        params = []
        for head in self.heads:
            gamma, beta = head(emb).chunk(2, dim=-1)
            params.append((gamma[..., None, None], beta[..., None, None]))
        return params


class FiLMMedVAE(nn.Module):
    """MedVAE + FiLM : forward(x, c) → (reconstruction, posterior)."""

    def __init__(self, base: AutoencoderKL):
        super().__init__()
        self.ae = base
        self._blocks = list(self._resnet_blocks())
        self.film = FiLMConditioner([b.conv2.out_channels for b in self._blocks])

    def _resnet_blocks(self):
        enc, dec = self.ae.encoder, self.ae.decoder
        for i_level in range(enc.num_resolutions):
            for i_block in range(enc.num_res_blocks):
                yield enc.down[i_level].block[i_block]
        yield enc.mid.block_1
        yield enc.mid.block_2
        yield dec.mid.block_1
        yield dec.mid.block_2
        for i_level in reversed(range(dec.num_resolutions)):
            for i_block in range(dec.num_res_blocks + 1):
                yield dec.up[i_level].block[i_block]

    def forward(self, x, c, sample_posterior=True):
        hooks = [block.register_forward_hook(lambda mod, inp, out, g=g, b=b: (1 + g) * out + b)
                 for block, (g, b) in zip(self._blocks, self.film(c))]
        try:
            posterior = self.ae.encode(x)
            z = posterior.sample() if sample_posterior else posterior.mode()
            return self.ae.decode(z), posterior
        finally:
            for hook in hooks:
                hook.remove()


class PlainMedVAE(nn.Module):
    """MedVAE sans conditionnement, même interface forward(x, c) (c ignoré)."""

    def __init__(self, base: AutoencoderKL):
        super().__init__()
        self.ae = base

    def forward(self, x, c=None, sample_posterior=True):
        posterior = self.ae.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.ae.decode(z), posterior
