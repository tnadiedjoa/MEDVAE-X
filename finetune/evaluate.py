"""Réévalue le checkpoint d'un run terminé sur le test set, avec les métriques actuelles.

Utile quand les métriques évoluent (ex. ajout du Dice sur les artères seules) : le
modèle est reconstruit depuis le config.yaml du run, son checkpoint rechargé, et les
scores de results.json sont mis à jour (le Dice doit rester identique).

Usage (depuis la racine du repo) :
    python -m finetune.evaluate experiments/runs/<run> [experiments/runs/<run> ...]
"""

import argparse
import json
import os

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from finetune.config import load_config
from finetune.dataset import ArcadeDataset
from finetune.metrics import SegMetrics


class AStar(torch.nn.Module):
    """Comme eval_astar : MedVAE en précision normale, U-Net sous autocast."""

    def __init__(self, autoencoder, unet):
        super().__init__()
        self.autoencoder, self.unet = autoencoder, unet

    def forward(self, x):
        with autocast(device_type=x.device.type, enabled=False):
            reconstructed = self.autoencoder(x.float())
        return self.unet(reconstructed)


def build_eval_model(config: dict, run_dir: str, device: torch.device) -> torch.nn.Module:
    """Modèle du run avec ses poids ; A* = U-Net de A appliqué aux images reconstruites."""
    if "checkpoint_a" in config["experiment"]:
        from finetune.encoder import MedVAEAutoencoder
        from finetune.models import build_unet

        unet = build_unet(config["model"])
        unet.load_state_dict(torch.load(config["experiment"]["checkpoint_a"], map_location=device)["model"])
        return AStar(MedVAEAutoencoder(device=device), unet).to(device)

    from finetune.train import build_model

    model = build_model(config, device)
    ckpt_path = os.path.join(run_dir, f"best_model_{config['experiment']['name']}.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    return model.to(device)


def evaluate_run(run_dir: str, device: torch.device) -> dict:
    config = load_config(os.path.join(run_dir, "config.yaml"))
    model = build_eval_model(config, run_dir, device).eval()

    data_cfg = config["data"]
    loader = DataLoader(
        ArcadeDataset(data_cfg["val_images"], data_cfg["val_ann"], augment=False),
        batch_size=4, shuffle=False, num_workers=data_cfg["num_workers"],
    )
    metrics = SegMetrics(num_classes=data_cfg["num_classes"], device=device)
    with torch.no_grad():
        for images, masks in loader:
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images.to(device))
            metrics.update(logits, masks.to(device))
    return metrics.compute()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_dir in args.runs:
        results_path = os.path.join(run_dir, "results.json")
        with open(results_path) as f:
            all_results = json.load(f)
        (condition, previous), = all_results.items()

        r = evaluate_run(run_dir, device)
        print(f"{os.path.basename(os.path.normpath(run_dir))} : Dice {previous['dice_mean']:.4f} -> "
              f"{r['dice_mean']:.4f} | Dice artères {r['dice_fg_mean']:.4f} | IoU artères {r['iou_fg_mean']:.4f}")
        all_results[condition] = {
            **previous,
            **{k: r[k] for k in ("dice_mean", "iou_mean", "dice_fg_mean", "iou_fg_mean",
                                 "dice_per_class", "iou_per_class")},
        }
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
