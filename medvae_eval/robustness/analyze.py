"""Analyse du sweep de robustesse (sweep.py) : moyennes par niveau et IC 95 % bootstrap.

Pour chaque dégradation et chaque niveau : PSNR(propre, dégradée), PSNR(propre,
reconstruite) et leur différence delta, sur toute l'image et sur les vaisseaux.
L'intervalle de confiance est obtenu par bootstrap sur les images (appariées).

Usage : python medvae_eval/robustness/analyze.py [--csv .../sweep.csv]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "medvae_eval" / "outputs" / "robustness" / "sweep.csv"
PARAM_LABEL = {"poisson": "Poisson scale (lower = noisier)", "jpeg": "JPEG quality",
               "blur": "Gaussian blur kernel size"}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, level), g in df.groupby(["degradation", "level"]):
        row = {"degradation": kind, "level": level, "param": g["param"].iloc[0], "n_images": len(g)}
        for col in ("psnr_clean_deg", "psnr_clean_rec", "psnr_deg_rec", "delta",
                    "mpsnr_clean_deg", "mpsnr_clean_rec", "mpsnr_deg_rec", "mdelta"):
            vals = g[col].to_numpy()
            vals = vals[np.isfinite(vals)]   # PSNR infini : entrée identique (flou de noyau 1)
            if len(vals) == 0:
                row.update({col: np.nan, f"{col}_lo": np.nan, f"{col}_hi": np.nan})
                continue
            lo, hi = bootstrap_ci(vals)
            row.update({col: vals.mean(), f"{col}_lo": lo, f"{col}_hi": hi})
        # Part des images pour lesquelles la reconstruction est plus proche de l'image propre
        row["frac_restored"] = float((g["delta"] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, baseline: float, out_png: Path, prefix: str, title: str):
    kinds = ["poisson", "jpeg", "blur"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    for j, kind in enumerate(kinds):
        s = summary[summary["degradation"] == kind].sort_values("level")
        x = s["param"].to_numpy()
        ax = axes[0, j]
        for col, label, color in ((f"{prefix}psnr_clean_deg", "degraded input vs clean", "#888888"),
                                  (f"{prefix}psnr_clean_rec", "MedVAE reconstruction vs clean", "#1f77b4"),
                                  (f"{prefix}psnr_deg_rec", "reconstruction vs degraded input\n(original study's metric)", "#ff7f0e")):
            ax.plot(x, s[col], marker="o", ms=3, label=label, color=color)
            ax.fill_between(x, s[f"{col}_lo"], s[f"{col}_hi"], alpha=0.2, color=color)
        ax.axhline(baseline, ls=":", color="k", lw=1, label="reconstruction of clean image")
        ax.set_title(kind)
        ax.set_ylabel("PSNR (dB)")
        if j == 0:
            ax.legend(fontsize=7)
        ax = axes[1, j]
        d = f"{prefix}delta" if prefix else "delta"
        ax.plot(x, s[d], marker="o", ms=3, color="#2ca02c")
        ax.fill_between(x, s[f"{d}_lo"], s[f"{d}_hi"], alpha=0.25, color="#2ca02c")
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel("Δ PSNR (dB)  > 0 : MedVAE restores")
        ax.set_xlabel(PARAM_LABEL[kind])
        if kind == "poisson":
            axes[0, j].invert_xaxis()   # axe partagé par la colonne : inverser une seule fois
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    summary = summarize(df)
    out_dir = Path(args.csv).parent
    summary.to_csv(out_dir / "summary.csv", index=False)

    base = df.drop_duplicates("image")
    plot(summary, base["psnr_clean_recclean"].mean(), out_dir / "robustness_full_image.png", "",
         f"MedVAE robustness — full image ({len(base)} seg_val images, 95% bootstrap CI)")
    plot(summary, base["mpsnr_clean_recclean"].mean(), out_dir / "robustness_vessels.png", "m",
         f"MedVAE robustness — vessel pixels ({len(base)} seg_val images, 95% bootstrap CI)")

    pd.set_option("display.width", 200)
    cols = ["degradation", "level", "param", "psnr_clean_deg", "psnr_clean_rec", "delta",
            "delta_lo", "delta_hi", "mdelta", "mdelta_lo", "mdelta_hi", "frac_restored"]
    print(summary[cols].round(2).to_string(index=False))
    print(f"\nReconstruction de l'image propre : PSNR {base['psnr_clean_recclean'].mean():.2f} dB, "
          f"vaisseaux {base['mpsnr_clean_recclean'].mean():.2f} dB")


if __name__ == "__main__":
    main()
