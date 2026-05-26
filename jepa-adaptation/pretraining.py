from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JEPA_ROOT    = Path(__file__).resolve().parent

if str(JEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(JEPA_ROOT))

from dataset.pretraining_dataset import get_pretraining_splits          # noqa: E402
from utils.jepa_trainer import JEPATrainer, load_yaml                   # noqa: E402
from utils.vae_trainer import Phase1Trainer, Phase2CLIPTrainer          # noqa: E402


def _resolve(path) -> Path | None:
    if path in (None, ""):
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _build_jepa_trainer_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    phase = cfg["phase2_jepa"]
    return {
        "training": {k: v for k, v in phase.items() if k not in ("jepa", "phase1_ckpt")},
        "model": {
            "autoencoder": cfg["model"],
            "jepa":        phase["jepa"],
        },
        "data": cfg["data"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-phase MedVAE pretraining on MedMNIST")
    p.add_argument(
        "--config",
        default=(JEPA_ROOT / "configs" / "pretraining.yaml").as_posix(),
        help="Path to pretraining config yaml",
    )
    p.add_argument(
        "--phase", type=int, choices=[1, 2], required=True,
        help="1 = VAE reconstruction pretraining  |  2 = fine-tuning",
    )
    p.add_argument(
        "--mode", choices=["jepa", "clip"], default="jepa",
        help="Phase-2 variant (ignored for phase 1): jepa or clip",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_yaml(args.config)

    mnst = cfg.get("medmnist", {})
    print(
        f"Building MedMNIST dataset "
        f"(data_fraction={mnst.get('data_fraction', 1.0)}, "
        f"phase1_ratio={mnst.get('phase1_ratio', 0.7)}, "
        f"size={mnst.get('size', 64)}, "
        f"as_rgb={mnst.get('as_rgb', False)}) …"
    )
    phase1_ds, phase2_ds = get_pretraining_splits(
        root          = mnst.get("root", "~/.medmnist"),
        data_fraction = float(mnst.get("data_fraction", 1.0)),
        phase1_ratio  = float(mnst.get("phase1_ratio", 0.7)),
        size          = int(mnst.get("size", 64)),
        as_rgb        = bool(mnst.get("as_rgb", False)),
        download      = bool(mnst.get("download", True)),
        seed          = int(mnst.get("seed", 42)),
    )
    print(f"  Phase-1 split : {len(phase1_ds):,} samples")
    print(f"  Phase-2 split : {len(phase2_ds):,} samples")

    val_ratio  = float(mnst.get("val_ratio", 0.1))
    n_val      = max(1, round(len(phase1_ds) * val_ratio))
    n_train    = len(phase1_ds) - n_val
    gen        = torch.Generator().manual_seed(int(mnst.get("seed", 42)) + 1)
    phase1_train_ds, phase1_val_ds = random_split(phase1_ds, [n_train, n_val], generator=gen)
    print(f"  Phase-1 train : {len(phase1_train_ds):,} | val: {len(phase1_val_ds):,}")

    if args.phase == 1:
        print("\n" + "=" * 60)
        print("Phase 1 — VAE reconstruction pretraining")
        print("=" * 60)
        trainer = Phase1Trainer(
            config        = {"training": cfg["phase1"], "model": cfg["model"], "data": cfg["data"]},
            train_dataset = phase1_train_ds,
            val_dataset   = phase1_val_ds,
        )
        trainer.fit()

    elif args.phase == 2 and args.mode == "clip":
        print("\n" + "=" * 60)
        print("Phase 2 (CLIP) — BiomedCLIP latent alignment fine-tuning")
        print("=" * 60)
        phase2_clip = cfg["phase2_clip"]
        trainer = Phase2CLIPTrainer(
            config        = {"training": phase2_clip, "model": cfg["model"], "data": cfg["data"]},
            train_dataset = phase2_ds,
            phase1_ckpt   = _resolve(phase2_clip.get("phase1_ckpt")),
        )
        trainer.fit()

    elif args.phase == 2 and args.mode == "jepa":
        print("\n" + "=" * 60)
        print("Phase 2 (JEPA) — latent-prediction pretraining")
        print("=" * 60)
        phase2_jepa = cfg["phase2_jepa"]
        trainer = JEPATrainer(
            config        = _build_jepa_trainer_config(cfg),
            train_dataset = phase2_ds,
            phase1_ckpt   = _resolve(phase2_jepa.get("phase1_ckpt")),
        )
        trainer.fit()

    else:
        raise ValueError(f"Unknown combination: --phase {args.phase} --mode {args.mode}")


if __name__ == "__main__":
    main()
