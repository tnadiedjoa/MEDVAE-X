"""Visibilité des vaisseaux fins et épais (E20) : rapport contraste sur bruit (CNR).

Complément de E11 (PSNR) : mêmes images (les n premières de seg_val), mêmes dégradations et
même bruit seedé que sweep.py et baselines.py. Pour chaque version d'une image (propre,
reconstruction MedVAE de la propre ; puis, à chaque niveau de dégradation : dégradée, MedVAE
de la dégradée, filtres gaussien et médian 5×5 de la dégradée), on mesure

    CNR = (moyenne du fond local − moyenne sur la ligne centrale des vaisseaux) / écart-type du fond local

Les vaisseaux sont sombres : le CNR est positif, et d'autant plus grand que les vaisseaux se
détachent du fond. La géométrie vient des annotations de l'image propre, identique pour toutes
les versions :
    - ligne centrale : squelette du masque ; largeur locale = 2 × distance au bord du masque ;
    - vaisseaux fins : largeur annotée ≤ 6 px (quartile inférieur des 100 images) ;
      épais : ≥ 12 px (quartile supérieur) ; « all » : tous ;
    - fond local : pixels hors du masque dilaté de 3 px et à moins de 13 px du masque, hors du
      bord noir du collimateur, rattachés au point de ligne centrale le plus proche (donc à la
      classe fine ou épaisse de ce point).

Le CNR récompense le lissage (le bruit du fond baisse) : il se lit avec le mPSNR de E11, et
séparément sur les vaisseaux fins, que le lissage efface. Les colonnes contrast et noise donnent
le numérateur et le dénominateur.

Usage (depuis medvae_eval/robustness/ ; le calcul complet prend ~13 min sur une RTX 3090) :
    python cnr.py --out ../../experiments/robustness/e20_cnr
ou en deux moitiés, puis fusion :
    python cnr.py --start 0 --n-images 50 --out <dossier>/part0
    python cnr.py --start 50 --n-images 50 --out <dossier>/part1
    python cnr.py --from-csv <dossier>/part0/cnr.csv <dossier>/part1/cnr.csv --out <dossier>
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm

from sweep import SEG_VAL, Reconstructor, degradation_params, degrade, vessel_masks

REPO_ROOT = Path(__file__).resolve().parents[2]
THIN_MAX, THICK_MIN = 6, 12        # largeur annotée (px)
GAP, RING = 3, 10                  # fond : entre 3 et 3 + 10 px du masque
FILTERS = {
    "gauss5": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "median5": lambda img: cv2.medianBlur(img, 5),
}
METHODS = ("degraded", "medvae", "gauss5", "median5")
KINDS = {"poisson": "Poisson noise", "jpeg": "JPEG compression", "blur": "Gaussian blur"}


def geometry(mask: np.ndarray, clean_u8: np.ndarray) -> dict:
    """{classe: (pixels de ligne centrale, pixels de fond local)} pour thin, thick et all."""
    m = mask.astype(np.uint8)
    skel = skeletonize(mask)
    width = 2 * cv2.distanceTransform(m, cv2.DIST_L2, 5)
    # pour chaque pixel, largeur du point de ligne centrale le plus proche
    _, labels = cv2.distanceTransformWithLabels(np.where(skel, 0, 1).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(labels.max() + 1, np.float32)
    lut[labels[skel]] = width[skel]
    nearest_width = lut[labels]
    dist_to_mask = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    background = (dist_to_mask > GAP) & (dist_to_mask <= GAP + RING) & (clean_u8 > 15)   # hors collimateur
    select = {"thin": nearest_width <= THIN_MAX, "thick": nearest_width >= THICK_MIN,
              "all": np.ones_like(mask, bool)}
    return {c: (skel & s, background & s) for c, s in select.items()}


def measure(img: np.ndarray, geo: dict) -> dict:
    """img dans [0, 1] → {classe: (cnr, contraste, bruit)} (NaN si trop peu de pixels)."""
    out = {}
    for c, (centre, bg) in geo.items():
        if centre.sum() < 20 or bg.sum() < 100:
            out[c] = (np.nan, np.nan, np.nan)
            continue
        b = img[bg]
        contrast, noise = float(b.mean() - img[centre].mean()), float(b.std())
        out[c] = (contrast / noise, contrast, noise)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame, out: Path):
    """Moyennes par réglage, écarts appariés par image (IC bootstrap sur les images)."""
    wide = df.pivot_table(index=["image", "degradation", "level", "cls"], columns="method", values="cnr")
    rows = []
    for (kind, level, cls), g in wide.groupby(level=["degradation", "level", "cls"]):
        row = {"degradation": kind, "level": level, "cls": cls}
        for m in g.columns:
            row[m] = float(g[m].mean())
        for a, b in (("medvae", "degraded"), ("gauss5", "degraded"), ("medvae", "gauss5"), ("medvae", "median5")):
            if a in g and b in g:
                v = (g[a] - g[b]).dropna().to_numpy()
                if len(v):
                    lo, hi = bootstrap_ci(v)
                    row[f"{a}-{b}"], row[f"{a}-{b}_ci"] = float(v.mean()), f"[{lo:+.2f}, {hi:+.2f}]"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    clean = df[df["degradation"] == "none"].pivot_table(index="cls", columns="method", values="cnr")
    lines = ["# E20 — CNR des vaisseaux (moyenne sur les images)", "",
             "Images propres : " + " ; ".join(f"{c} : propre {clean.loc[c, 'clean']:.2f}, MedVAE {clean.loc[c, 'medvae']:.2f}"
                                              for c in ("thin", "thick", "all")), ""]
    for cls in ("thin", "thick", "all"):
        lines += [f"## Vaisseaux {cls}", "",
                  "| Dégradation | Niveau | dégradée | MedVAE | gauss5 | median5 | MedVAE − gauss5 [IC 95 %] |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in summary[summary["cls"] == cls].iterrows():
            if r["degradation"] == "none":
                continue
            lines.append(f"| {r['degradation']} | {r['level']} | {r['degraded']:.2f} | {r['medvae']:.2f} | "
                         f"{r['gauss5']:.2f} | {r['median5']:.2f} | {r['medvae-gauss5']:+.2f} {r['medvae-gauss5_ci']} |")
        lines.append("")

    # Mécanisme : contraste conservé (fraction du contraste de l'image propre, par image) et bruit du fond
    ref = df[(df["degradation"] == "none") & (df["method"] == "clean")].set_index(["image", "cls"])
    deg = df[df["degradation"] != "none"].join(ref[["contrast", "noise"]], on=["image", "cls"], rsuffix="_clean")
    deg = deg.assign(contrast_kept=deg["contrast"] / deg["contrast_clean"])
    mech = deg.groupby(["degradation", "level", "cls", "method"])[["contrast_kept", "noise"]].mean().reset_index()
    mech.to_csv(out / "mechanism.csv", index=False)
    clean_noise = ref.groupby("cls")["noise"].mean()
    lines += ["## Mécanisme : contraste conservé (fraction du contraste propre) / bruit du fond", "",
              "Bruit du fond sur l'image propre : " + " ; ".join(f"{c} {clean_noise[c]:.3f}" for c in ("thin", "thick")), "",
              "| Dégradation | Niveau | Vaisseaux | dégradée | MedVAE | gauss5 | median5 |", "|---|---|---|---|---|---|---|"]
    for (kind, level, cls), g in mech.groupby(["degradation", "level", "cls"]):
        if level not in (0, 3, 6, 9) or cls == "all":
            continue
        g = g.set_index("method")
        cells = [f"{g.loc[m, 'contrast_kept']:.2f} / {g.loc[m, 'noise']:.3f}" for m in METHODS]
        lines.append(f"| {kind} | {level} | {cls} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines))


def plot(df: pd.DataFrame, out: Path):
    colors = {"degraded": "#7f7f7f", "medvae": "#d62728", "gauss5": "#1f77b4", "median5": "#2ca02c"}
    labels = {"degraded": "degraded input", "medvae": "MedVAE", "gauss5": "Gaussian 5×5", "median5": "median 5×5"}
    clean = df[df["degradation"] == "none"].groupby(["cls", "method"])["cnr"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey="row")
    for i, cls in enumerate(("thin", "thick")):
        for j, kind in enumerate(KINDS):
            ax = axes[i, j]
            g = df[(df["degradation"] == kind) & (df["cls"] == cls)].groupby(["method", "level"])["cnr"].mean()
            for m in METHODS:
                ax.plot(g[m].index, g[m].values, marker="o", ms=3, color=colors[m], label=labels[m])
            ax.axhline(clean[(cls, "clean")], color="black", ls=":", lw=1, label="clean image")
            ax.axhline(clean[(cls, "medvae")], color=colors["medvae"], ls=":", lw=1, label="MedVAE, clean input")
            ax.set_title(f"{KINDS[kind]} — {cls} vessels", fontsize=10)
            ax.set_xlabel("degradation level (0 = weakest, 9 = strongest)", fontsize=8.5)
            ax.grid(alpha=0.25)
        axes[i, 0].set_ylabel(f"CNR, {cls} vessels")
    axes[0, 0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "cnr.pdf")
    fig.savefig(out / "cnr.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=0, help="première image (pour découper le calcul)")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "robustness" / "e20_cnr"))
    parser.add_argument("--from-csv", nargs="+", default=None,
                        help="ne rien recalculer : fusionner ces cnr.csv, puis résumé et figure dans --out")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        df = pd.concat([pd.read_csv(f) for f in args.from_csv], ignore_index=True)
        df.to_csv(out / "cnr.csv", index=False)
        summarize(df, out)
        plot(df, out)
        print(f"{df['image'].nunique()} images -> {out}/cnr.csv, summary.csv, summary.md, cnr.png")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = vessel_masks(SEG_VAL / "annotations" / "seg_val.json")
    files = sorted(masks)[args.start: args.start + args.n_images]
    recon = Reconstructor(device)
    params = degradation_params(args.levels)

    rows = []
    for idx, name in enumerate(tqdm(files, desc="images"), start=args.start):   # idx global : même bruit que sweep.py
        clean_u8 = cv2.imread(str(SEG_VAL / "images" / name), cv2.IMREAD_GRAYSCALE)
        geo = geometry(masks[name], clean_u8)
        versions = {("none", 0, None, "clean"): clean_u8 / 255.0,
                    ("none", 0, None, "medvae"): recon([clean_u8])[0]}
        for kind, values in params.items():
            degraded = []
            for level, value in enumerate(values):
                rng = np.random.default_rng([args.seed, idx, level])   # même bruit que sweep.py
                degraded.append(degrade(clean_u8, kind, value, rng))
            for level, (value, d_u8, r) in enumerate(zip(values, degraded, recon(degraded))):
                versions[(kind, level, float(value), "degraded")] = d_u8 / 255.0
                versions[(kind, level, float(value), "medvae")] = r
                for fname, f in FILTERS.items():
                    versions[(kind, level, float(value), fname)] = f(d_u8) / 255.0
        for (kind, level, value, method), img in versions.items():
            for cls, (c, contrast, noise) in measure(img, geo).items():
                rows.append({"image": name, "degradation": kind, "level": level, "param": value, "method": method,
                             "cls": cls, "cnr": c, "contrast": contrast, "noise": noise})

    df = pd.DataFrame(rows)
    df.to_csv(out / "cnr.csv", index=False)
    summarize(df, out)
    plot(df, out)
    print((out / "summary.md").read_text())
    print(f"-> {out}/cnr.csv, summary.csv, summary.md, cnr.png")


if __name__ == "__main__":
    main()
