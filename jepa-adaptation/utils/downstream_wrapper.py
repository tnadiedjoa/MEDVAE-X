from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

JEPA_ROOT = Path(__file__).resolve().parents[1]
if str(JEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(JEPA_ROOT))

from models import AutoencoderKL2D, MedVAE_JEPA
from utils.jepa_trainer import build_model, load_yaml


def _resolve(path: Optional[str | Path]) -> Optional[Path]:
    if path in (None, ""):
        return None
    path = Path(path)
    return path if path.is_absolute() else JEPA_ROOT / path


def _remap_lora_state_dict(sd: dict) -> dict:
    remapped = {}
    for k, v in sd.items():
        if ".lora_down." in k or ".lora_up." in k:
            continue
        new_k = k.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")
        remapped[new_k] = v
    return remapped


class EncoderWrapper(nn.Module):
    latent_channels: int

    @torch.no_grad()
    def encode(self, _x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MedVAEBaselineEncoder(EncoderWrapper):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.autoencoder = AutoencoderKL2D(
            ddconfig=config["ddconfig"],
            embed_dim=int(config.get("embed_dim", 3)),
            ckpt_path=None,  # load manually to handle LoRA key remapping
            apply_channel_ds=bool(config.get("apply_channel_ds", False)),
        )
        ckpt_path = _resolve(config.get("ckpt_path"))
        if ckpt_path is not None:
            raw = torch.load(ckpt_path, map_location="cpu")
            sd = raw.get("state_dict", raw) if bool(config.get("state_dict", True)) else raw
            sd = _remap_lora_state_dict(sd)
            missing, unexpected = self.autoencoder.load_state_dict(sd, strict=False)
            print(f"MedVAE loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        self.autoencoder.requires_grad_(False)
        self.latent_channels = int(config.get("embed_dim", 3))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.autoencoder.encode(x).mode()


class JEPAAdaptedEncoder(EncoderWrapper):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        ckpt_path = _resolve(config["ckpt_path"])
        model_cfg = load_yaml(_resolve(config.get("model_config", "configs/model.yaml")))

        model: MedVAE_JEPA = build_model(model_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt))

        self.autoencoder = model.context_encoder.autoencoder
        self.autoencoder.requires_grad_(False)
        self.latent_channels = int(model_cfg["autoencoder"]["embed_dim"])

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x).mode()
