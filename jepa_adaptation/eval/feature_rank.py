"""Dimensionnalité des features entraînées par JEPA (axe C) : y a-t-il contraction ?

Sur les 300 images de test, on prend les features de l'encodeur avant conv_out (la couche
que la loss JEPA entraîne, ramenée à 64×64, comme le linear probe), un vecteur par pixel,
standardisé canal par canal. Sur un échantillon de pixels, on calcule :
    - le rang effectif exp(H(p)), p_i = σ_i / Σσ (σ : valeurs singulières) ;
    - le nombre de composantes principales pour 90 % de la variance.
Une contraction de l'espace (JEPA qui se concentre sur quelques directions) fait baisser
les deux.

Usage (racine du repo) :
    python jepa_adaptation/eval/feature_rank.py --model stage2 --ckpt jepa_adaptation/outputs/stage2/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from finetune.config import load_config    # noqa: E402
from finetune.dataset import ArcadeDataset  # noqa: E402
from probe import build_encoder, extract    # noqa: E402


def spectrum_stats(x: np.ndarray) -> dict:
    """x : (n, c) vecteurs standardisés → rang effectif et dimensions pour 90 % de variance."""
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = s / s.sum()
    var = np.cumsum(s ** 2) / (s ** 2).sum()
    return {"effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
            "dims_90pct_variance": int(np.searchsorted(var, 0.90)) + 1,
            "n_channels": int(x.shape[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["medvae", "stage1", "stage2"], required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n-pixels", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.model != "medvae" and not args.ckpt:
        parser.error("--ckpt est requis pour stage1 / stage2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_config(str(REPO_ROOT / "finetune" / "configs" / "condition_a.yaml"))["data"]
    feats, _ = extract(build_encoder(args.model, args.ckpt, device),
                       ArcadeDataset(d["val_images"], d["val_ann"]), device, "penultimate")
    x = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
    idx = torch.from_numpy(np.random.default_rng(args.seed).choice(len(x), args.n_pixels, replace=False))
    x = x[idx].float().numpy()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)

    result = {"model": args.model, "ckpt": args.ckpt, "n_pixels": args.n_pixels, **spectrum_stats(x)}
    print(json.dumps(result, indent=2))
    out = Path(args.out or REPO_ROOT / "experiments" / "jepa" / f"feature_rank_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
