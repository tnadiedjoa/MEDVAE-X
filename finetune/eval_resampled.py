"""Témoin de A* (E16) : le U-Net de A appliqué à une image réduite puis ré-agrandie.

A* mesure ce que coûte à A le passage de l'image par MedVAE (latent 128×128, 16 fois moins
de valeurs). Ce témoin applique au même U-Net, sans réentraînement, une compression triviale
du même taux : moyenne sur des blocs 4×4 (image 128×128), puis ré-agrandissement à 512×512
(bilinéaire ou bicubique). Test set de 300 images, Dice / IoU sur les artères.

Usage (racine du repo) :
    python finetune/eval_resampled.py --run-a experiments/runs/*_e17_seed4?_condition_a \
        --out experiments/diagnostics/e16_resampled.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finetune.config import load_config               # noqa: E402
from finetune.dataset import ArcadeDataset             # noqa: E402
from finetune.evaluate import build_eval_model         # noqa: E402
from finetune.metrics import SegMetrics                # noqa: E402


class Resampled(torch.nn.Module):
    def __init__(self, unet, factor: int = 4, mode: str = "bilinear"):
        super().__init__()
        self.unet, self.factor, self.mode = unet, factor, mode

    def forward(self, x):
        small = F.avg_pool2d(x, self.factor)
        up = F.interpolate(small, scale_factor=self.factor, mode=self.mode, align_corners=False)
        return self.unet(up.clamp(0, 1))


@torch.no_grad()
def evaluate_run(run_a: str, device) -> list:
    config = load_config(str(Path(run_a) / "config.yaml"))
    unet = build_eval_model(config, run_a, device).eval()
    models = {"A": unet,
              "A, 128x128 then bilinear": Resampled(unet, mode="bilinear"),
              "A, 128x128 then bicubic": Resampled(unet, mode="bicubic")}
    data_cfg = config["data"]
    loader = DataLoader(ArcadeDataset(data_cfg["val_images"], data_cfg["val_ann"], augment=False),
                        batch_size=4, shuffle=False, num_workers=data_cfg["num_workers"])
    metrics = {name: SegMetrics(num_classes=data_cfg["num_classes"], device=device) for name in models}
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for name, model in models.items():
                metrics[name].update(model(images), masks)
    rows = []
    for name, m in metrics.items():
        r = m.compute()
        rows.append({"seed": config["experiment"].get("seed"), "run_a": Path(run_a).name, "model": name,
                     "dice_fg_mean": r["dice_fg_mean"], "iou_fg_mean": r["iou_fg_mean"],
                     "dice_mean": r["dice_mean"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", nargs="+", required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "diagnostics" / "e16_resampled.csv"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [row for run in args.run_a for row in evaluate_run(run, device)]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    summary = df.groupby("model")[["dice_fg_mean", "iou_fg_mean"]].agg(["mean", "std"])
    print(summary.round(4).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
