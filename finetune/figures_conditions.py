"""Schéma des cinq conditions de segmentation (A, A*, B, C, D), pour le rapport et le README.

Les vignettes viennent d'une vraie image du test set : l'angiographie, son latent MedVAE,
sa reconstruction et son masque de vérité terrain (26 classes).

Usage (racine du repo) : python -m finetune.figures_conditions
    → final_report/figures/theo/conditions.pdf (+ .png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from finetune.config import REPO_ROOT, load_config
from finetune.dataset import ArcadeDataset
from finetune.figures_report import mask_to_rgb

OUT = REPO_ROOT / "final_report" / "figures" / "theo" / "conditions.pdf"
TRAINED, FROZEN = "#3d7ab8", "#dbe8f5"


@torch.no_grad()
def thumbnails(index: int = 144):   # image du test set contrastée, 9 segments annotés
    from medvae import MVAE
    d = load_config(str(REPO_ROOT / "finetune" / "configs" / "condition_a.yaml"))["data"]
    image, mask = ArcadeDataset(d["val_images"], d["val_ann"])[index]
    ae = MVAE(model_name="medvae_4_1_2d", modality="xray").eval().model
    z = ae.encode(image[None] * 2 - 1).mode()
    rec = ((ae.decode(z) + 1) / 2).clamp(0, 1)
    return image[0].numpy(), z[0, 0].numpy(), rec[0, 0].numpy(), mask.numpy()


def box(ax, x, y, w, h, title, sub, trained):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=TRAINED if trained else FROZEN,
                                edgecolor=TRAINED, linewidth=1.2, linestyle="-" if trained else "--"))
    color = "white" if trained else "#1d3d5c"
    ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=6.6, color=color, linespacing=0.95)


def thumb(ax, x, y, s, img, caption, cmap="gray", **kw):
    ax.imshow(img, cmap=cmap, extent=(x, x + s, y, y + s), interpolation="nearest", **kw)
    ax.text(x + s / 2, y - 0.08, caption, ha="center", va="top", fontsize=6.5, style="italic", color="#444444")


def arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=10, linewidth=1.1,
                                 color="#333333"))


def main():
    image, latent, rec, mask = thumbnails()
    mask_rgb = mask_to_rgb(mask)

    s, gap = 0.95, 1.55   # taille des vignettes, hauteur d'une ligne
    rows = [
        ("A", "Full-resolution\nbaseline",
         [("img", image, "image 512²"), ("box", "U-Net (ResNet-34)", "trained on original images", True),
          ("img", mask_rgb, "26-class mask")]),
        ("A*", "Compression\nalone (control)",
         [("img", image, "image 512²"), ("box", "MedVAE", "encode → decode, frozen", False),
          ("img", rec, "reconstruction 512²"), ("box", "U-Net of A", "frozen, no retraining", False),
          ("img", mask_rgb, "26-class mask")]),
        ("B", "Segmentation\nfrom the latent",
         [("img", image, "image 512²"), ("box", "MedVAE encoder", "pre-trained, frozen", False),
          ("img", latent, "latent 128²×1"), ("box", "Latent U-Net head", "U-Net at 128²,\nthen ×4 upsampling", True),
          ("img", mask_rgb, "26-class mask")]),
        ("C", "Latent of a\nfine-tuned MedVAE",
         [("img", image, "image 512²"), ("box", "MedVAE encoder", "fine-tuned on ARCADE,\nfrozen", False),
          ("img", latent, "latent 128²×1"), ("box", "Latent U-Net head", "U-Net at 128²,\nthen ×4 upsampling", True),
          ("img", mask_rgb, "26-class mask")]),
        ("D", "Segmentation of\nreconstructions",
         [("img", image, "image 512²"), ("box", "MedVAE", "encode → decode, frozen", False),
          ("img", rec, "reconstruction 512²"), ("box", "U-Net (ResNet-34)", "trained on reconstructions", True),
          ("img", mask_rgb, "26-class mask")]),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 1.2 * len(rows) + 0.4))
    ax.set_xlim(-1.45, 10.6)
    ax.set_ylim(-0.6, gap * len(rows) - 0.25)
    ax.axis("off")
    for r, (name, title, items) in enumerate(rows):
        y = gap * (len(rows) - 1 - r)
        ax.text(-1.4, y + s / 2 + 0.12, name, fontsize=13, fontweight="bold", va="center")
        ax.text(-1.4, y + s / 2 - 0.3, title, fontsize=6.8, va="center")
        # positions : vignettes de largeur s, boîtes de largeur 2.1, flèches entre les deux
        if len(items) == 3:
            xs = [0.4, 2.9, 9.4]
            widths = [s, 5.0, s]
        else:
            xs = [0.4, 1.95, 4.65, 6.2, 9.4]
            widths = [s, 2.1, s, 2.6, s]
        for (kind, *payload), x, w in zip(items, xs, widths):
            if kind == "img":
                if payload[0] is latent:   # contraste sur l'intérieur (le bord du latent est saturé)
                    lo, hi = np.percentile(latent[8:-8, 8:-8], [1, 99])
                    thumb(ax, x, y, s, latent, payload[1], cmap="viridis", vmin=lo, vmax=hi)
                else:
                    thumb(ax, x, y, s, payload[0], payload[1])
            else:
                box(ax, x, y + 0.12, w, s - 0.24, *payload)
        for (x1, w1), x2 in zip(zip(xs[:-1], widths[:-1]), xs[1:]):
            arrow(ax, x1 + w1 + 0.05, x2 - 0.05, y + s / 2)
    ax.text(10.6, -0.5, "dark: trained   ·   light, dashed: frozen", ha="right", fontsize=7, color="#444444")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
