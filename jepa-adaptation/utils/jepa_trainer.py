from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JEPA_ROOT = Path(__file__).resolve().parents[1]
MEDVAE_ROOT = PROJECT_ROOT / "MedVAE-main"


def ensure_medvae_on_path() -> None:
    if MEDVAE_ROOT.exists() and str(MEDVAE_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDVAE_ROOT))
    if str(JEPA_ROOT) not in sys.path:
        sys.path.insert(0, str(JEPA_ROOT))


ensure_medvae_on_path()

from dataset.dataset import ArcadeDataset, get_files_labels  # noqa: E402
from models import AutoencoderKL2D, AutoencoderKL3D, MedVAE_JEPA  # noqa: E402


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_training_config(
    training_config: str | Path,
    model_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    cfg = load_yaml(training_config)
    if model_config is not None:
        cfg = deep_update(cfg, {"model": load_yaml(model_config)})
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: Optional[str | Path]) -> Optional[Path]:
    if path in (None, ""):
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_autoencoder(config: Dict[str, Any]) -> torch.nn.Module:
    spatial_dims = int(config.get("spatial_dims", 2))
    ddconfig = config["ddconfig"]
    embed_dim = int(config["embed_dim"])
    ckpt_path = resolve_path(config.get("ckpt_path"))
    ignore_keys = config.get("ignore_keys", [])
    apply_channel_ds = bool(config.get("apply_channel_ds", True))
    state_dict = bool(config.get("state_dict", True))

    if spatial_dims == 2:
        return AutoencoderKL2D(
            ddconfig=ddconfig,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            apply_channel_ds=apply_channel_ds,
            state_dict=state_dict,
        )

    if spatial_dims == 3:
        return AutoencoderKL3D(
            ddconfig=ddconfig,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            apply_channel_ds=apply_channel_ds,
        )

    raise ValueError("model.spatial_dims must be 2 or 3.")


def build_model(config: Dict[str, Any]) -> MedVAE_JEPA:
    autoencoder = build_autoencoder(config["autoencoder"])
    jepa_config = config["jepa"]
    return MedVAE_JEPA(
        autoencoder=autoencoder,
        latent_dim=int(jepa_config["latent_dim"]),
        patch_size=jepa_config.get("patch_size", 16),
        target_mask_ratio=float(jepa_config.get("target_mask_ratio", 0.4)),
        predictor_hidden_dim=jepa_config.get("predictor_hidden_dim"),
        predictor_depth=int(jepa_config.get("predictor_depth", 3)),
        spatial_dims=int(config["autoencoder"].get("spatial_dims", 2)),
        ema_momentum=float(jepa_config.get("ema_momentum", 0.996)),
        loss_type=jepa_config.get("loss_type", "smooth_l1"),
        normalize_loss=bool(jepa_config.get("normalize_loss", True)),
        sample_posterior=bool(jepa_config.get("sample_posterior", False)),
        freeze_decoder=bool(jepa_config.get("freeze_decoder", True)),
    )


def build_dataloader(split_config: Dict[str, Any], loader_config: Dict[str, Any]):
    data_dir = resolve_path(split_config["data_dir"])
    files, labels = get_files_labels(data_dir.as_posix())
    dataset = ArcadeDataset(
        files,
        labels=labels,
        return_path=bool(split_config.get("return_path", False)),
        num_classes=int(split_config.get("num_classes", 26)),
    )
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 4)),
        shuffle=bool(split_config.get("shuffle", False)),
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
        drop_last=bool(split_config.get("drop_last", False)),
    )


def batch_to_image(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        for key in ("image", "images", "x"):
            if key in batch:
                return batch[key]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


class JEPATrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        train_cfg = config["training"]
        device_name = train_cfg.get("device", "auto")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        set_seed(int(train_cfg.get("seed", 42)))

        self.model = build_model(config["model"]).to(self.device)
        self.optimizer = self.build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        )

        loader_cfg = config["data"].get("loader", {})
        self.train_loader = build_dataloader(config["data"]["train"], loader_cfg)
        valid_cfg = config["data"].get("valid")
        self.valid_loader = (
            build_dataloader(valid_cfg, loader_cfg) if valid_cfg is not None else None
        )

        self.output_dir = resolve_path(train_cfg.get("output_dir", "outputs/jepa"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_valid_loss = float("inf")

    def build_optimizer(self):
        opt_cfg = self.config["training"].get("optimizer", {})
        name = opt_cfg.get("name", "adamw").lower()
        lr = float(opt_cfg.get("lr", 1e-4))
        weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
        params = self.model.trainable_parameters()

        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer: {name}")

    def save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float]):
        path = self.output_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        return path

    def train_step(self, batch: Any) -> Dict[str, float]:
        self.model.train()
        images = batch_to_image(batch).to(self.device, non_blocking=True).float()
        self.optimizer.zero_grad(set_to_none=True)

        amp_enabled = self.scaler.is_enabled()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = self.model(images, return_loss=True)
            loss = outputs["loss"]

        self.scaler.scale(loss).backward()
        max_norm = self.config["training"].get("max_grad_norm")
        if max_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), float(max_norm)
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.update_target_encoder()

        logs = outputs.get("logs", {})
        metrics = {key: float(value.detach().cpu()) for key, value in logs.items()}
        metrics["loss"] = float(loss.detach().cpu())
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.valid_loader is None:
            return {}

        self.model.eval()
        totals: Dict[str, float] = {}
        count = 0
        for batch in self.valid_loader:
            images = batch_to_image(batch).to(self.device, non_blocking=True).float()
            outputs = self.model(images, return_loss=True)
            logs = outputs.get("logs", {})
            metrics = {key: float(value.detach().cpu()) for key, value in logs.items()}
            metrics["loss"] = float(outputs["loss"].detach().cpu())
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1

        return {key: value / max(count, 1) for key, value in totals.items()}

    def fit(self):
        train_cfg = self.config["training"]
        epochs = int(train_cfg.get("epochs", 1))
        log_every = int(train_cfg.get("log_every", 20))
        save_every = int(train_cfg.get("save_every", 1))
        global_step = 0

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for step, batch in enumerate(self.train_loader, start=1):
                metrics = self.train_step(batch)
                running_loss += metrics["loss"]
                global_step += 1

                if step % log_every == 0:
                    avg_loss = running_loss / log_every
                    running_loss = 0.0
                    print(
                        f"epoch={epoch} step={step} global_step={global_step} "
                        f"loss={avg_loss:.5f}"
                    )

            valid_metrics = self.validate()
            if valid_metrics:
                print(
                    f"epoch={epoch} valid_loss={valid_metrics['loss']:.5f}"
                )
                if valid_metrics["loss"] < self.best_valid_loss:
                    self.best_valid_loss = valid_metrics["loss"]
                    self.save_checkpoint("best.pt", epoch, valid_metrics)

            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, valid_metrics)

        self.save_checkpoint("last.pt", epochs, {"best_valid_loss": self.best_valid_loss})


def parse_args():
    parser = argparse.ArgumentParser(description="Train JEPA adaptation for Med-VAE.")
    parser.add_argument(
        "--config",
        default=(JEPA_ROOT / "configs" / "training.yaml").as_posix(),
        help="Path to training yaml.",
    )
    parser.add_argument(
        "--model-config",
        default=(JEPA_ROOT / "configs" / "model.yaml").as_posix(),
        help="Path to model yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_training_config(args.config, args.model_config)
    trainer = JEPATrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
