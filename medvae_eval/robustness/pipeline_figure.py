"""Schéma du protocole de robustesse corrigé (figure du rapport et des slides).

Usage : python medvae_eval/robustness/pipeline_figure.py
        → final_report/figures/elias/pipeline_corrected.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "final_report" / "figures" / "elias" / "pipeline_corrected.pdf"

BLUE, GREY, ORANGE, GREEN = "#dbe8f5", "#ececec", "#fbe3c8", "#d9efd9"


def box(ax, x, y, w, h, text, color, size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="#444444", linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.3, color="#444444", linestyle="--" if dashed else "-"))


def main():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    # Chaîne de traitement
    box(ax, 0.1, 2.3, 1.6, 0.9, "Clean image\n$x$", GREY, bold=True)
    box(ax, 2.3, 2.3, 1.9, 0.9, "Degradation\nPoisson / JPEG / blur\n$N$ levels", BLUE, size=9)
    box(ax, 4.8, 2.3, 1.6, 0.9, "Degraded\n$d$", GREY, bold=True)
    box(ax, 6.9, 2.3, 1.2, 0.9, "MedVAE\n(mean latent)", BLUE, size=9)
    box(ax, 8.5, 2.3, 1.4, 0.9, "Reconstruction\n$r$", GREY, bold=True, size=9)
    for x1, x2 in ((1.7, 2.3), (4.2, 4.8), (6.4, 6.9), (8.1, 8.5)):
        arrow(ax, x1, 2.75, x2, 2.75)

    # Mesures, toutes par rapport à l'image propre
    box(ax, 1.6, 0.95, 2.6, 0.75, "PSNR$(x, d)$\ndegradation amplitude", ORANGE, size=9)
    box(ax, 5.8, 0.95, 2.6, 0.75, "PSNR$(x, r)$\ndistance to the clean image", ORANGE, size=9)
    arrow(ax, 0.9, 2.3, 2.3, 1.7, dashed=True)
    arrow(ax, 5.6, 2.3, 3.6, 1.7, dashed=True)
    arrow(ax, 0.9, 2.3, 6.1, 1.7, dashed=True)
    arrow(ax, 9.2, 2.3, 8.1, 1.7, dashed=True)

    box(ax, 2.5, 0.05, 5.0, 0.6,
        r"$\Delta = $ PSNR$(x, r)$ $-$ PSNR$(x, d)$   ( > 0 : MedVAE restores )", GREEN, size=9, bold=True)
    arrow(ax, 2.9, 0.95, 4.2, 0.65)
    arrow(ax, 7.1, 0.95, 5.8, 0.65)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=110, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
