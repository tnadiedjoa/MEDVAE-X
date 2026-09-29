"""Figures de l'axe C (rapport) avec les modèles E19 (seed 42) : reconstructions et PCA des features.

    - recon.png : une image de test et ses reconstructions (MedVAE, étape 1, étape 2 JEPA) ;
    - pca.png   : features avant conv_out projetées sur 3 composantes principales (RGB),
                  PCA ajustée sur 50 images de test, montrée sur 3 autres.

Usage (racine du repo) : python jepa_adaptation/eval/figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from finetune.config import load_config               # noqa: E402
from finetune.dataset import ArcadeDataset             # noqa: E402
from probe import _penultimate_hook, load_autoencoder  # noqa: E402

OUT = REPO_ROOT / "final_report" / "figures" / "yanic"
RUNS = REPO_ROOT / "experiments" / "runs"                                       # pré-entraînement E19, seed 42
MODELS = {"MedVAE": ("medvae", None),
          "Stage 1": ("stage1", str(sorted(RUNS.glob("*_e19_jepa_stage1_seed42"))[-1] / "best.pt")),
          "Stage 2 (JEPA)": ("stage2", str(sorted(RUNS.glob("*_e19_jepa_stage2_seed42"))[-1] / "best.pt"))}
SHOW = (144, 120, 219)


@torch.no_grad()
def penultimate(ae, x, batch_size=2):
    feats = []
    for chunk in x.split(batch_size):
        store = []
        handle = _penultimate_hook(ae.encoder, store)
        try:
            ae.encode(chunk)
        finally:
            handle.remove()
        feats.append(store[0].float().cpu())
    return torch.cat(feats)


def pca_rgb(fit: torch.Tensor, show: torch.Tensor) -> np.ndarray:
    c = fit.shape[1]
    pca = PCA(n_components=3).fit(fit.permute(0, 2, 3, 1).reshape(-1, c).numpy()[::7])
    proj = pca.transform(show.permute(0, 2, 3, 1).reshape(-1, c).numpy())
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    proj = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return proj.reshape(show.shape[0], show.shape[2], show.shape[3], 3)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_config(str(REPO_ROOT / "finetune" / "configs" / "condition_a.yaml"))["data"]
    ds = ArcadeDataset(d["val_images"], d["val_ann"])
    show = torch.stack([ds[i][0] for i in SHOW]).to(device) * 2 - 1
    fit_idx = [i for i in range(0, 300, 6) if i not in SHOW]                      # images non montrées
    fit = torch.stack([ds[i][0] for i in fit_idx]).to(device) * 2 - 1

    recons, rgbs = {}, {}
    for label, (model, ckpt) in MODELS.items():
        ae = load_autoencoder(model, ckpt, device)
        rec = ae.decode(ae.encode(show[:1]).mode())
        recons[label] = ((rec[0, 0].float().cpu() + 1) / 2).clamp(0, 1).numpy()
        rgbs[label] = pca_rgb(penultimate(ae, fit), penultimate(ae, show))
        del ae
        torch.cuda.empty_cache()

    x0 = ((show[0, 0].cpu() + 1) / 2).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    axes[0].imshow(x0, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=9)
    for ax, (label, rec) in zip(axes[1:], recons.items()):
        psnr = -10 * np.log10(np.mean((rec - x0) ** 2))
        ax.imshow(rec, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}\nPSNR {psnr:.1f} dB", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "recon.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(len(SHOW), 4, figsize=(8.4, 2.1 * len(SHOW)))
    for r in range(len(SHOW)):
        axes[r, 0].imshow(((show[r, 0].cpu() + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
        for c, (label, rgb) in enumerate(rgbs.items(), start=1):
            axes[r, c].imshow(rgb[r])
            if r == 0:
                axes[r, c].set_title(label, fontsize=9)
        if r == 0:
            axes[r, 0].set_title("image", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "pca.png", dpi=140)
    plt.close(fig)
    print(f"-> {OUT}/recon.png, {OUT}/pca.png")


if __name__ == "__main__":
    main()
