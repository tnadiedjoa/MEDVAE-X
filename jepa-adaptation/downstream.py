from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

JEPA_ROOT = Path(__file__).resolve().parent
if str(JEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(JEPA_ROOT))

from dataset.dataloaders import get_loaders
from utils.downstream_wrapper import JEPAAdaptedEncoder, MedVAEBaselineEncoder
from utils.jepa_trainer import load_yaml


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, bg_weight: float = 0.1, dice_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        weights = torch.ones(num_classes + 1)
        weights[0] = bg_weight
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weights)

        probs = torch.softmax(logits, dim=1)
        dice = 0.0
        for c in range(1, self.num_classes + 1):
            p = probs[:, c]
            t = (target == c).float()
            intersection = (p * t).sum()
            dice += 1.0 - (2.0 * intersection + 1.0) / (p.sum() + t.sum() + 1.0)
        dice = dice / self.num_classes

        return ce + self.dice_weight * dice


class SegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_size: int = 256, stages: int = 4):
        super().__init__()
        self.image_size = image_size
        widths = [in_channels, 128, 64, 32, 16]
        while len(widths) <= stages:
            widths.append(widths[-1])

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(widths[i], widths[i + 1], 3, padding=1),
                nn.BatchNorm2d(widths[i + 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(stages)
        ])
        self.classifier = nn.Conv2d(widths[stages], num_classes + 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.classifier(x)


def resize_batch(
    images: torch.Tensor, masks: torch.Tensor, size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    masks = (
        F.interpolate(masks.unsqueeze(1).float(), size=(size, size), mode="nearest")
        .squeeze(1)
        .long()
    )
    return images, masks


def _make_seg_cmap(num_colors: int):
    colors = [(0, 0, 0)]
    base = plt.get_cmap("tab20", num_colors - 1)
    for i in range(num_colors - 1):
        colors.append(base(i))
    return mcolors.ListedColormap(colors)


def compute_metrics(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    dice_sum = iou_sum = 0.0
    valid = 0

    for c in range(1, num_classes + 1):
        p = pred == c
        t = target == c
        intersection = (p & t).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()

        if t_sum == 0 and p_sum == 0:
            continue
        valid += 1
        dice_sum += 2 * intersection / (p_sum + t_sum + 1e-8)
        iou_sum += intersection / ((p_sum + t_sum - intersection) + 1e-8)

    if valid == 0:
        return {"dice": 0.0, "iou": 0.0}
    return {"dice": dice_sum / valid, "iou": iou_sum / valid}


class DownstreamTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        name: str,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.encoder = encoder.to(device).eval()
        self.name = name
        self.device = device
        self.cfg = config
        self.image_size = int(config.get("image_size", 256))
        self.num_classes = int(config.get("num_classes", 26))
        self.log_every = int(config.get("log_every", 20))

        self.head = SegHead(
            in_channels=encoder.latent_channels,
            num_classes=self.num_classes,
            image_size=self.image_size,
            stages=int(config.get("decoder_stages", 4)),
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(config.get("epochs", 20)),
            eta_min=float(config.get("lr", 1e-3)) * 0.01,
        )
        self.criterion = DiceCELoss(
            num_classes=self.num_classes,
            bg_weight=float(config.get("bg_weight", 0.1)),
            dice_weight=float(config.get("dice_weight", 0.5)),
        ).to(device)

        output_dir = Path(config.get("output_dir", "outputs/downstream"))
        self.viz_dir = output_dir / self.name.replace(" ", "_")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self._cmap = _make_seg_cmap(self.num_classes + 1)

    def _save_viz(self, val_loader, epoch: int) -> None:
        self.head.eval()
        with torch.no_grad():
            images, masks = next(iter(val_loader))
            images, masks = resize_batch(images, masks, self.image_size)
            images = images.to(self.device)
            z = self.encoder.encode(images)
            logits = self.head(z)
            preds = logits.argmax(dim=1)

        n = min(4, images.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None]

        for i in range(n):
            img = images[i, 0].cpu().numpy()
            gt = masks[i].cpu().numpy()
            pr = preds[i].cpu().numpy()

            axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title("Input" if i == 0 else "")
            axes[i, 1].imshow(gt, cmap=self._cmap, vmin=0, vmax=self.num_classes, interpolation="nearest")
            axes[i, 1].set_title("Ground truth" if i == 0 else "")
            axes[i, 2].imshow(pr, cmap=self._cmap, vmin=0, vmax=self.num_classes, interpolation="nearest")
            axes[i, 2].set_title("Prediction" if i == 0 else "")

            for ax in axes[i]:
                ax.axis("off")

        fig.suptitle(f"{self.name} — epoch {epoch}", fontsize=11)
        fig.tight_layout()
        path = self.viz_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def _train_epoch(self, loader, epoch: int) -> float:
        self.head.train()
        total_loss = 0.0

        for step, (images, masks) in enumerate(loader, start=1):
            images, masks = resize_batch(images, masks, self.image_size)
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.no_grad():
                z = self.encoder.encode(images)

            logits = self.head(z)
            loss = self.criterion(logits, masks)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if step % self.log_every == 0:
                avg = total_loss / step
                print(f"  [{self.name}] epoch={epoch} step={step} loss={avg:.4f}")

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _evaluate(self, loader) -> Dict[str, float]:
        self.head.eval()
        dice_sum = iou_sum = 0.0
        count = 0

        for images, masks in loader:
            images, masks = resize_batch(images, masks, self.image_size)
            images = images.to(self.device)
            masks = masks.to(self.device)

            z = self.encoder.encode(images)
            logits = self.head(z)
            m = compute_metrics(logits, masks, self.num_classes)
            dice_sum += m["dice"]
            iou_sum += m["iou"]
            count += 1

        n = max(count, 1)
        return {"dice": dice_sum / n, "iou": iou_sum / n}

    def fit(self, train_loader, val_loader) -> Dict[str, float]:
        epochs = int(self.cfg.get("epochs", 20))
        best: Dict[str, float] = {"dice": 0.0, "iou": 0.0}

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            metrics = self._evaluate(val_loader)
            self._save_viz(val_loader, epoch)
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            print(
                f"[{self.name}] epoch={epoch}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_dice={metrics['dice']:.4f}  val_iou={metrics['iou']:.4f}  "
                f"lr={lr:.2e}"
            )
            if metrics["dice"] > best["dice"]:
                best = metrics

        return best


def _build_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream segmentation: MedVAE baseline vs JEPA-adapted"
    )
    parser.add_argument(
        "--config",
        default=(JEPA_ROOT / "configs" / "downstream.yaml").as_posix(),
        help="Path to downstream config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    ds_cfg = cfg["downstream"]

    device = _build_device(ds_cfg.get("device", "auto"))
    print(f"Device: {device}")

    print("\nLoading arcade dataset...")
    loaders = get_loaders(
        batch_size=int(ds_cfg.get("batch_size", 4)),
        num_workers=int(ds_cfg.get("num_workers", 0)),
    )
    train_loader = loaders["stenosis_train"]
    val_loader = loaders["stenosis_val"]

    results: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 60)
    print("Training downstream head on MedVAE baseline (pre-trained)")
    print("=" * 60)
    medvae_enc = MedVAEBaselineEncoder(cfg["medvae"])
    medvae_trainer = DownstreamTrainer(medvae_enc, "MedVAE-baseline", ds_cfg, device)
    results["MedVAE-baseline"] = medvae_trainer.fit(train_loader, val_loader)

    print("\n" + "=" * 60)
    print("Training downstream head on JEPA-adapted encoder")
    print("=" * 60)
    jepa_enc = JEPAAdaptedEncoder(cfg["jepa"])
    jepa_trainer = DownstreamTrainer(jepa_enc, "JEPA-adapted", ds_cfg, device)
    results["JEPA-adapted"] = jepa_trainer.fit(train_loader, val_loader)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (best validation metrics)")
    print("=" * 60)
    print(f"{'Model':<22} {'Dice':>10} {'IoU':>10}")
    print("-" * 44)
    for name, m in results.items():
        print(f"{name:<22} {m['dice']:>10.4f} {m['iou']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
