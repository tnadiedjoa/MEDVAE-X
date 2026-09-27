"""MedVAE rend-il la segmentation plus robuste aux dégradations ? (axe A, volet aval)

Sur le test set (300 images), chaque image est dégradée (bruit de Poisson, JPEG, flou,
mêmes plages que sweep.py), puis segmentée par :
    A      : U-Net entraîné sur les images propres, appliqué à l'image dégradée
    A*     : le même U-Net, appliqué à la reconstruction MedVAE de l'image dégradée
    D      : MedVAE → U-Net entraîné sur des reconstructions (condition D)
On rapporte le Dice sur les artères (fond exclu) à chaque niveau.

Usage (racine du repo) :
    python medvae_eval/robustness/downstream.py --run-a experiments/runs/<A> --run-d experiments/runs/<D>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune.config import load_config                      # noqa: E402
from finetune.dataset import ArcadeDataset                    # noqa: E402
from finetune.encoder import MedVAEAutoencoder                # noqa: E402
from finetune.evaluate import AStar, build_eval_model         # noqa: E402
from finetune.metrics import SegMetrics                       # noqa: E402
from sweep import degradation_params, degrade                 # noqa: E402


def load_model(run_dir: str, device):
    config = load_config(str(Path(run_dir) / "config.yaml"))
    return build_eval_model(config, run_dir, device).eval(), config


def degrade_batch(images: torch.Tensor, kind: str, value, seed: int, offset: int) -> torch.Tensor:
    """images [B,1,H,W] dans [0,1] → même format, dégradées (bruit seedé par image)."""
    out = []
    for i, img in enumerate(images):
        u8 = (img[0].numpy() * 255).round().astype(np.uint8)
        rng = np.random.default_rng([seed, offset + i])
        out.append(torch.from_numpy(degrade(u8, kind, value, rng).astype(np.float32) / 255.0))
    return torch.stack(out).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-d", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9],
                        help="indices de niveaux parmi une grille de 10 (0 = le plus faible)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None, help="test rapide")
    parser.add_argument("--out", default=str(REPO_ROOT / "medvae_eval" / "outputs" / "robustness" / "downstream.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, config = load_model(args.run_a, device)
    model_d, _ = load_model(args.run_d, device)
    model_astar = AStar(MedVAEAutoencoder(device=device), model_a).to(device).eval()
    models = {"A": model_a, "A*": model_astar, "D": model_d}

    data_cfg = config["data"]
    loader = DataLoader(ArcadeDataset(data_cfg["val_images"], data_cfg["val_ann"], augment=False),
                        batch_size=4, shuffle=False, num_workers=data_cfg["num_workers"])
    params = degradation_params(10)
    settings = [("none", 0, None)] + [(k, lvl, params[k][lvl]) for k in params for lvl in args.levels]

    rows = []
    for kind, level, value in tqdm(settings, desc="réglages"):
        metrics = {name: SegMetrics(num_classes=data_cfg["num_classes"], device=device) for name in models}
        offset = 0
        for b, (images, masks) in enumerate(loader):
            if args.max_batches is not None and b >= args.max_batches:
                break
            if kind != "none":
                images = degrade_batch(images, kind, value, args.seed, offset)
            offset += len(images)
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad(), autocast(device_type=device.type, dtype=torch.bfloat16,
                                           enabled=device.type == "cuda"):
                for name, model in models.items():
                    metrics[name].update(model(images), masks)
        for name, m in metrics.items():
            r = m.compute()
            rows.append({"degradation": kind, "level": level,
                         "param": None if value is None else float(value), "model": name,
                         "dice_fg_mean": r["dice_fg_mean"], "iou_fg_mean": r["iou_fg_mean"],
                         "dice_mean": r["dice_mean"]})
        print(pd.DataFrame(rows[-3:])[["degradation", "level", "model", "dice_fg_mean"]].to_string(index=False))

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
