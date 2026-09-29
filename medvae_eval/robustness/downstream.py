"""MedVAE rend-il la segmentation plus robuste aux dégradations ? (axe A, volet aval)

Sur le test set (300 images), chaque image est dégradée (bruit de Poisson, JPEG, flou,
mêmes plages que sweep.py), puis segmentée par :
    A      : U-Net entraîné sur les images propres, appliqué à l'image dégradée
    A*     : le même U-Net, appliqué à la reconstruction MedVAE de l'image dégradée
    D      : MedVAE → U-Net entraîné sur des reconstructions (condition D)
On rapporte le Dice sur les artères (fond exclu) à chaque niveau. Plusieurs paires de runs
(un A et un D par seed) donnent la variabilité d'un entraînement à l'autre ; les images
dégradées sont identiques pour toutes les paires.

Usage (racine du repo) :
    python medvae_eval/robustness/downstream.py \
        --run-a experiments/runs/*_e10_seed4?_condition_a --run-d experiments/runs/*_e10_seed4?_condition_d
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def run_seed(run_dir: str):
    return load_config(str(Path(run_dir) / "config.yaml"))["experiment"].get("seed")


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
    parser.add_argument("--run-a", nargs="+", required=True)
    parser.add_argument("--run-d", nargs="*", default=None,
                        help="un run D par run A, appariés par seed (optionnel : A et A* seulement)")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9],
                        help="indices de niveaux parmi une grille de 10 (0 = le plus faible)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None, help="test rapide")
    parser.add_argument("--out", default=str(REPO_ROOT / "medvae_eval" / "outputs" / "robustness" / "downstream.csv"))
    args = parser.parse_args()
    if args.run_d and len(args.run_a) != len(args.run_d):
        parser.error("autant de runs D que de runs A")
    # A et D appariés par seed d'entraînement (l'ordre des dossiers suit l'heure de lancement)
    runs_a = sorted(args.run_a, key=run_seed)
    runs_d = sorted(args.run_d, key=run_seed) if args.run_d else [None] * len(runs_a)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = degradation_params(10)
    settings = [("none", 0, None)] + [(k, lvl, params[k][lvl]) for k in params for lvl in args.levels]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_a, run_d in zip(runs_a, runs_d):
        rows += evaluate_pair(run_a, run_d, settings, args, device)
        pd.DataFrame(rows).to_csv(args.out, index=False)   # sauvegarde après chaque paire

    df = pd.DataFrame(rows)
    plot(df, Path(args.out).with_suffix(".png"))
    print(f"-> {args.out}")


def evaluate_pair(run_a, run_d, settings, args, device):
    model_a, config = load_model(run_a, device)
    seed = config["experiment"].get("seed")   # seed d'entraînement du run A
    model_astar = AStar(MedVAEAutoencoder(device=device), model_a).to(device).eval()
    models = {"A": model_a, "A*": model_astar}
    if run_d is not None:
        models["D"] = load_model(run_d, device)[0]

    data_cfg = config["data"]
    loader = DataLoader(ArcadeDataset(data_cfg["val_images"], data_cfg["val_ann"], augment=False),
                        batch_size=4, shuffle=False, num_workers=data_cfg["num_workers"])
    rows = []
    for kind, level, value in tqdm(settings, desc=f"réglages (seed {seed})"):
        torch.manual_seed(args.seed)   # latent MedVAE tiré au hasard : même tirage pour chaque réglage
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
            rows.append({"seed": seed, "run_a": Path(run_a).name, "run_d": run_d and Path(run_d).name,
                         "degradation": kind, "level": level,
                         "param": None if value is None else float(value), "model": name,
                         "dice_fg_mean": r["dice_fg_mean"], "iou_fg_mean": r["iou_fg_mean"],
                         "dice_mean": r["dice_mean"]})
        print(pd.DataFrame(rows[-len(models):])[["degradation", "level", "model", "dice_fg_mean"]].to_string(index=False))
    del models, model_a, model_astar
    torch.cuda.empty_cache()
    return rows


def plot(df: pd.DataFrame, out: Path):
    """Dice artères vs niveau de dégradation (moyenne ± écart-type entre seeds), un panneau par dégradation."""
    stats = df.groupby(["degradation", "level", "param", "model"], dropna=False)["dice_fg_mean"].agg(["mean", "std"])
    stats = stats.reset_index()
    clean = stats[stats["degradation"] == "none"].set_index("model")
    kinds = [k for k in stats["degradation"].unique() if k != "none"]
    colors = {"A": "#1f77b4", "A*": "#ff7f0e", "D": "#2ca02c"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.6), sharey=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        sub = stats[stats["degradation"] == kind]
        for model, color in colors.items():
            s = sub[sub["model"] == model].sort_values("level")
            if s.empty:
                continue
            ax.errorbar(s["level"], s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3,
                        color=color, label=model)
            ax.axhline(clean.loc[model, "mean"], color=color, linestyle=":", linewidth=1)
        ax.set_title(kind)
        ax.set_xlabel("niveau de dégradation (0 = le plus faible)")
        ax.set_xticks(sorted(sub["level"].unique()))
    np.atleast_1d(axes)[0].set_ylabel("Dice artères (test)")
    np.atleast_1d(axes)[0].legend(title="pointillés : image propre", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
