"""Schéma du PSNR sur les vaisseaux (mPSNR) pour le rapport et les slides.

Une image de seg_val et sa version bruitée (Poisson) partagent le même masque de vaisseaux ;
le mPSNR compare les deux images sur ces seuls pixels.

Usage (racine du repo) : python medvae_eval/robustness/mpsnr_figure.py
    → final_report/figures/elias/mpsnr.pdf
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from sweep import SEG_VAL, degrade, vessel_masks

OUT = Path(__file__).resolve().parents[2] / "final_report" / "figures" / "elias" / "mpsnr.pdf"


def overlay(img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.repeat(img_u8[..., None], 3, axis=2).astype(float) / 255
    rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.array([0.9, 0.1, 0.45])
    return rgb


def main(name: str = "1.png"):
    masks = vessel_masks(SEG_VAL / "annotations" / "seg_val.json")
    clean = cv2.imread(str(SEG_VAL / "images" / name), cv2.IMREAD_GRAYSCALE)
    noisy = degrade(clean, "poisson", 0.2, np.random.default_rng(0))
    mask = masks[name].astype(bool)

    fig = plt.figure(figsize=(5.6, 2.9))
    ax1 = fig.add_axes([0.0, 0.02, 0.4, 0.84])
    ax2 = fig.add_axes([0.6, 0.02, 0.4, 0.84])
    for ax, img, title in ((ax1, clean, "Clean image"), (ax2, noisy, "Degraded image (Poisson noise)")):
        ax.imshow(overlay(img, mask))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.patches.append(FancyArrowPatch((0.41, 0.5), (0.59, 0.5), transform=fig.transFigure,
                                       arrowstyle="<|-|>", mutation_scale=12, color="#b0105a", linewidth=1.5))
    fig.text(0.5, 0.36, "mPSNR\non vessel\npixels (pink)", ha="center", va="top", fontsize=8.5, color="#b0105a")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
