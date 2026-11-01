"""Does MedVAE make segmentation more robust to degradations? (part A, downstream)

On the test set (300 images), each image is degraded (Poisson noise, JPEG, blur, same
ranges as sweep.py), then segmented by:
    A      : U-Net trained on clean images, applied to the degraded image
    A*     : the same U-Net, applied to the MedVAE reconstruction of the degraded image
    D      : MedVAE → U-Net trained on reconstructions (condition D)
The Dice on the arteries (background excluded) is reported at each level. Several pairs of
runs (one A and one D per seed) give the run-to-run variability; the degraded images are
identical for all pairs.

Usage (from the repository root):
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
    """images [B,1,H,W] in [0,1] → same format, degraded (noise seeded per image)."""
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
                        help="one D run per A run, paired by seed (optional: A and A* only)")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9],
                        help="level indices in a grid of 10 (0 = the weakest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None, help="test rapide")
    parser.add_argument("--out", default=str(REPO_ROOT / "medvae_eval" / "outputs" / "robustness" / "downstream.csv"))
    args = parser.parse_args()
    if args.run_d and len(args.run_a) != len(args.run_d):
        parser.error("as many D runs as A runs")
    # A and D paired by training seed (the folder order follows the launch time)
    runs_a = sorted(args.run_a, key=run_seed)
    runs_d = sorted(args.run_d, key=run_seed) if args.run_d else [None] * len(runs_a)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = degradation_params(10)
    settings = [("none", 0, None)] + [(k, lvl, params[k][lvl]) for k in params for lvl in args.levels]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_a, run_d in zip(runs_a, runs_d):
        rows += evaluate_pair(run_a, run_d, settings, args, device)
        pd.DataFrame(rows).to_csv(args.out, index=False)   # save after each pair

    df = pd.DataFrame(rows)
    plot(df, Path(args.out).with_suffix(".png"))
    print(f"-> {args.out}")


def evaluate_pair(run_a, run_d, settings, args, device):
    model_a, config = load_model(run_a, device)
    seed = config["experiment"].get("seed")   # training seed of run A
    model_astar = AStar(MedVAEAutoencoder(device=device), model_a).to(device).eval()
    models = {"A": model_a, "A*": model_astar}
    if run_d is not None:
        models["D"] = load_model(run_d, device)[0]

    data_cfg = config["data"]
    loader = DataLoader(ArcadeDataset(data_cfg["val_images"], data_cfg["val_ann"], augment=False),
                        batch_size=4, shuffle=False, num_workers=data_cfg["num_workers"])
    rows = []
    for kind, level, value in tqdm(settings, desc=f"settings (seed {seed})"):
        torch.manual_seed(args.seed)   # MedVAE latent sampled at random: same draw for each setting
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
    """Artery Dice vs degradation level (mean ± standard deviation across seeds), one panel per degradation."""
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
        ax.set_xlabel("degradation level (0 = the weakest)")
        ax.set_xticks(sorted(sub["level"].unique()))
    np.atleast_1d(axes)[0].set_ylabel("Artery Dice (test)")
    np.atleast_1d(axes)[0].legend(title="dotted: clean image", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
