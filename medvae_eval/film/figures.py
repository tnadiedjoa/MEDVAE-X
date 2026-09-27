"""Figures de l'axe B refait (E13) pour le rapport et les slides.

    - pipeline_film_corrected.pdf : protocole (entraînement identique, contrôles sur c) ;
    - film_results.pdf : PSNR de test par modèle, et différences appariées par image.

Usage (racine du repo) : python medvae_eval/film/figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from finetune.runs import RUNS_DIR  # noqa: E402

OUT = REPO_ROOT / "final_report" / "figures" / "theophile"
BLUE, GREY, ORANGE, GREEN = "#dbe8f5", "#ececec", "#fbe3c8", "#d9efd9"


def box(ax, x, y, w, h, text, color, size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="#444444", linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.2, color="#444444", linestyle="--" if dashed else "-"))


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    box(ax, 0.05, 2.2, 1.9, 0.95, "seg_train\n900 train / 100 val\n256×256", GREY, bold=True)
    box(ax, 2.5, 2.55, 2.6, 0.7, "Baseline: MedVAE\n(all weights, 3000 steps)", BLUE)
    box(ax, 2.5, 1.65, 2.6, 0.7, "FiLM-MedVAE($c$)\n(same training)", BLUE)
    box(ax, 0.05, 1.0, 1.9, 0.75, "quality score $c$\n(approach A, B or C)", ORANGE)
    arrow(ax, 1.95, 2.8, 2.5, 2.9)
    arrow(ax, 1.95, 2.6, 2.5, 2.0)
    arrow(ax, 1.95, 1.4, 2.5, 1.85, dashed=True)

    box(ax, 5.7, 2.2, 1.9, 0.95, "Test: seg_val\n200 unseen images", GREY, bold=True)
    arrow(ax, 5.1, 2.9, 5.7, 2.75)
    arrow(ax, 5.1, 2.0, 5.7, 2.5)

    box(ax, 8.1, 2.65, 1.85, 0.55, "real $c$", GREEN, bold=True)
    box(ax, 8.1, 1.95, 1.85, 0.55, "shuffled $c$\n(another image's)", GREEN)
    box(ax, 8.1, 1.25, 1.85, 0.55, "constant $c$\n(training mean)", GREEN)
    for y in (2.92, 2.22, 1.52):
        arrow(ax, 7.6, 2.65, 8.1, y)

    ax.text(5.0, 0.45, "PSNR, vessel PSNR, SSIM, HaarPSI  ·  3 seeds  ·  paired per-image differences\n"
            "If the model uses $c$: real $c$ > shuffled $c$ and real $c$ > constant $c$",
            ha="center", va="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_film_corrected.pdf", bbox_inches="tight")
    fig.savefig(OUT / "pipeline_film_corrected.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def load(pattern: str) -> list:
    return [pd.read_csv(r / "test_per_image.csv") for r in sorted(RUNS_DIR.glob(pattern))
            if (r / "test_per_image.csv").exists()]


def per_image(runs: list, variant: str, metric: str = "psnr") -> pd.Series:
    """PSNR par image, moyenné sur les seeds."""
    return pd.concat([df[df["variant"] == variant].set_index("image")[metric] for df in runs], axis=1).mean(axis=1)


def results():
    pre, base, film = load("*_e13_film_pretrained"), load("*_e13_film_baseline_seed4?"), load("*_e13_film_C_seed4?")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})

    # Niveaux (moyenne ± écart-type entre seeds)
    bars = [("pretrained", pre, "real_c"), ("baseline", base, "real_c"), ("FiLM\nreal $c$", film, "real_c"),
            ("FiLM\nshuffled $c$", film, "shuffled_c"), ("FiLM\nconstant $c$", film, "constant_c")]
    means = [np.mean([df[df["variant"] == v]["psnr"].mean() for df in runs]) for _, runs, v in bars]
    stds = [np.std([df[df["variant"] == v]["psnr"].mean() for df in runs], ddof=1) if len(runs) > 1 else 0
            for _, runs, v in bars]
    colors = ["#bbbbbb", "#1f77b4", "#2ca02c", "#98df8a", "#98df8a"]
    ax1.bar(range(len(bars)), means, yerr=stds, color=colors, capsize=3)
    ax1.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5)
    ax1.set_ylim(33, 43.5)
    ax1.set_ylabel("test PSNR (dB)")
    for i, m in enumerate(means):
        ax1.text(i, m + 0.15, f"{m:.2f}", ha="center", fontsize=8)
    ax1.set_title("Reconstruction on seg_val (256×256)", fontsize=10)

    # Différences appariées par image
    diffs = {"FiLM real $c$\n− baseline": per_image(film, "real_c") - per_image(base, "real_c"),
             "real $c$\n− shuffled $c$": per_image(film, "real_c") - per_image(film, "shuffled_c"),
             "real $c$\n− constant $c$": per_image(film, "real_c") - per_image(film, "constant_c")}
    ax2.boxplot(list(diffs.values()), showfliers=False, widths=0.5)
    for i, d in enumerate(diffs.values(), start=1):
        ax2.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d,
                    s=4, alpha=0.35, color="#555555")
        ax2.text(i, 0.97, f"mean {d.mean():+.3f}", ha="center", va="top", fontsize=8,
                 transform=ax2.get_xaxis_transform())
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(range(1, len(diffs) + 1), list(diffs), fontsize=8.5)
    ax2.set_ylabel("per-image PSNR difference (dB)")
    ax2.set_title("Paired differences (200 images, 3-seed mean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "film_results.pdf", bbox_inches="tight")
    fig.savefig(OUT / "film_results.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    results()
    print(f"-> {OUT}/pipeline_film_corrected.pdf, {OUT}/film_results.pdf")
