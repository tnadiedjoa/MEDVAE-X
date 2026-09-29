"""Figures de la section segmentation du rapport, à partir des runs E17 et du témoin R de E16
(3 seeds par condition ; A* : seed 42).

    - seg_results.pdf : Dice artères de test par condition (moyenne ± écart-type, un point par seed)
      et Dice par classe d'artère (moyenne sur les seeds) ;
    - seg_curves.pdf  : Dice de validation au fil des epochs (moyenne sur les seeds).

Usage (racine du repo) : python -m finetune.figures_report
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from finetune.config import REPO_ROOT

RUNS = REPO_ROOT / "experiments" / "runs"
OUT = REPO_ROOT / "final_report" / "figures" / "theo"
CONDITIONS = {"A": "e17_seed4?_condition_a", "A*": "e17_seed42_condition_astar", "B": "e17_seed4?_condition_b",
              "C": "e17_seed4?_condition_c", "D": "e17_seed4?_condition_d", "R": "e16_seed4?_condition_r"}
COLORS = {"A": "#1f77b4", "A*": "#aec7e8", "B": "#ff7f0e", "C": "#d62728", "D": "#2ca02c", "R": "#7f7f7f"}
ABSENT = {12}   # classe absente du test set (E09)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Masque de classes (0-25) → image RGB : fond sombre, une couleur par segment."""
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3].tolist()
    colors += plt.get_cmap("Set2")(np.linspace(0, 1, 8))[:, :3].tolist()
    rgb = np.ones((*mask.shape, 3)) * 0.12
    for c in range(1, 26):
        rgb[mask == c] = colors[c - 1]
    return rgb


def load() -> dict:
    """{condition: [(résultats, historique ou None), ...]} sur les seeds disponibles."""
    out = {}
    for name, pattern in CONDITIONS.items():
        runs = []
        for run in sorted(glob.glob(str(RUNS / f"*_{pattern}"))):
            results = next(iter(json.load(open(Path(run) / "results.json")).values()))
            hist = next(iter(Path(run).glob("*_history.json")), None)
            runs.append((results, json.load(open(hist)) if hist else None))
        out[name] = runs
    return out


def results_figure(data: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 3]})
    names = [n for n in CONDITIONS if data[n]]
    for i, n in enumerate(names):
        vals = [r["dice_fg_mean"] for r, _ in data[n]]
        ax1.bar(i, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0, color=COLORS[n], capsize=3)
        ax1.scatter(np.full(len(vals), i), vals, color="black", s=8, zorder=3)
        ax1.text(i, np.mean(vals) + 0.012, f"{np.mean(vals):.3f}", ha="center", fontsize=8)
    ax1.set_xticks(range(len(names)), names)
    ax1.set_ylim(0, 0.5)
    ax1.set_ylabel("artery Dice (test set)")
    ax1.set_title("Test Dice (3 seeds; A*: seed 42)", fontsize=10)

    classes = [c for c in range(1, 26) if c not in ABSENT]
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        per_class = np.mean([r["dice_per_class"] for r, _ in data[n]], axis=0)
        ax2.bar(np.arange(len(classes)) + (i - (len(names) - 1) / 2) * width, per_class[classes], width,
                color=COLORS[n], label=n)
    ax2.set_xticks(range(len(classes)), classes, fontsize=8)
    ax2.set_xlabel("artery segment (ARCADE class; class 12 absent from the test set)", fontsize=8.5)
    ax2.set_ylabel("Dice (mean over seeds)")
    ax2.legend(ncol=6, fontsize=8, loc="upper right")
    ax2.set_title("Per-class test Dice", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "seg_results.pdf")
    fig.savefig(OUT / "seg_results.png", dpi=140)
    plt.close(fig)


def curves_figure(data: dict):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for n in ("A", "B", "C", "D", "R"):
        hists = [h for _, h in data[n] if h]
        length = min(len(h["val"]) for h in hists)
        dice = np.mean([[e["dice_mean"] for e in h["val"][:length]] for h in hists], axis=0)
        ax.plot(range(1, length + 1), dice, color=COLORS[n], label=n)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Dice (seg_val, with background)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "seg_curves.pdf")
    fig.savefig(OUT / "seg_curves.png", dpi=140)
    plt.close(fig)


@torch.no_grad()
def predictions_figure(indices=(144, 120, 219)):
    """Prédictions des modèles du seed 42 sur quelques images du test set."""
    from torch.amp import autocast

    from finetune.config import load_config
    from finetune.dataset import ArcadeDataset
    from finetune.evaluate import build_eval_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_config(str(REPO_ROOT / "finetune" / "configs" / "condition_a.yaml"))["data"]
    dataset = ArcadeDataset(d["val_images"], d["val_ann"])
    samples = [dataset[i] for i in indices]
    images = torch.stack([x for x, _ in samples]).to(device)
    preds = {}
    for name, pattern in CONDITIONS.items():
        run = sorted(glob.glob(str(RUNS / f"*_{pattern.replace('seed4?', 'seed42')}")))[-1]
        model = build_eval_model(load_config(str(Path(run) / "config.yaml")), run, device).eval()
        torch.manual_seed(0)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            preds[name] = torch.cat([model(x[None]) for x in images]).argmax(1).cpu().numpy()
        del model
        torch.cuda.empty_cache()

    cols = ["image", "ground truth", *CONDITIONS]
    fig, axes = plt.subplots(len(indices), len(cols), figsize=(1.5 * len(cols), 1.55 * len(indices)))
    for r, (x, m) in enumerate(samples):
        panels = [x[0].numpy(), mask_to_rgb(m.numpy())] + [mask_to_rgb(preds[n][r]) for n in CONDITIONS]
        for c, (ax, panel) in enumerate(zip(axes[r], panels)):
            ax.imshow(panel, cmap="gray" if c == 0 else None)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "seg_predictions.png", dpi=150)
    plt.close(fig)


def main():
    data = load()
    results_figure(data)
    curves_figure(data)
    predictions_figure()
    print(f"-> {OUT}/seg_results.pdf, {OUT}/seg_curves.pdf, {OUT}/seg_predictions.png")


if __name__ == "__main__":
    main()
