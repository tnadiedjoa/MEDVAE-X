"""Témoins de l'axe A : que font de simples filtres, là où MedVAE « restaure » ?

Mêmes images (les n premières de seg_val), mêmes dégradations et même bruit seedé que
sweep.py ; au lieu de passer l'image dégradée d dans MedVAE, on la filtre :
    gauss5   flou gaussien 5×5 (σ déduit du noyau, OpenCV)
    median5  filtre médian 5×5
et on calcule le même gain delta = PSNR(propre, filtrée) − PSNR(propre, d), sur toute
l'image et sur les vaisseaux. Le tableau résumé met en regard le delta de MedVAE (E11).

Usage (racine du repo) :
    python medvae_eval/robustness/baselines.py --out experiments/robustness/e11_baselines
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sweep import SEG_VAL, degradation_params, degrade, psnr, vessel_masks

REPO_ROOT = Path(__file__).resolve().parents[2]
FILTERS = {
    "gauss5": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "median5": lambda img: cv2.medianBlur(img, 5),
}


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medvae-sweep", default=str(REPO_ROOT / "experiments" / "robustness" / "e11_sweep" / "sweep.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "robustness" / "e11_baselines"))
    args = parser.parse_args()

    masks = vessel_masks(SEG_VAL / "annotations" / "seg_val.json")
    files = sorted(masks)[: args.n_images]
    params = degradation_params(args.levels)
    rows = []
    for idx, name in enumerate(tqdm(files, desc="images")):
        clean_u8 = cv2.imread(str(SEG_VAL / "images" / name), cv2.IMREAD_GRAYSCALE)
        clean, mask = clean_u8 / 255.0, masks[name]
        for kind, values in params.items():
            for level, value in enumerate(values):
                rng = np.random.default_rng([args.seed, idx, level])   # même bruit que sweep.py
                d_u8 = degrade(clean_u8, kind, value, rng)
                d = d_u8 / 255.0
                for fname, f in FILTERS.items():
                    r = f(d_u8) / 255.0
                    rows.append({"image": name, "degradation": kind, "level": level, "param": float(value),
                                 "method": fname,
                                 "delta": psnr(clean, r) - psnr(clean, d),
                                 "mdelta": psnr(clean, r, mask) - psnr(clean, d, mask)})
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    medvae = pd.read_csv(args.medvae_sweep).assign(method="medvae")
    medvae = medvae[medvae["image"].isin(files)][["image", "degradation", "level", "param", "method", "delta", "mdelta"]]
    both = pd.concat([df, medvae]).replace([np.inf, -np.inf], np.nan)

    summary = []
    for (kind, level, method), g in both.groupby(["degradation", "level", "method"]):
        for col in ("delta", "mdelta"):
            v = g[col].dropna().to_numpy()
            if len(v):
                lo, hi = bootstrap_ci(v)
                summary.append({"degradation": kind, "level": level, "method": method, "metric": col,
                                "mean": float(v.mean()), "ci_low": lo, "ci_high": hi, "n": len(v)})
    summary = pd.DataFrame(summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "baselines.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    table = summary[summary["metric"] == "delta"].pivot_table(index=["degradation", "level"], columns="method",
                                                               values="mean")
    print(table.round(2).to_string())
    print(f"-> {out}/baselines.csv, {out}/summary.csv")


if __name__ == "__main__":
    main()
