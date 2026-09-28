"""Tableaux et figures de E12 (segmentation d'images dégradées, sorties de downstream.py).

Chaque courbe est une paire (fichier CSV, modèle) : Dice artères moyen ± écart-type entre
seeds, en fonction du niveau de dégradation ; le trait horizontal pointillé est le Dice
sur les images propres.

Usage (racine du repo) :
    python medvae_eval/robustness/analyze_downstream.py \
        --curve "A=experiments/robustness/e12_downstream/downstream.csv:A" \
        --curve "D=experiments/robustness/e12_downstream/downstream.csv:D" \
        --out experiments/robustness/e12_downstream/main
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

KINDS = {"poisson": "Poisson noise", "jpeg": "JPEG compression", "blur": "Gaussian blur"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f", "#9467bd", "#d62728"]


def load_curves(specs: list) -> pd.DataFrame:
    frames = []
    for spec in specs:
        label, rest = spec.split("=", 1)
        csv, model = rest.rsplit(":", 1)
        df = pd.read_csv(csv)
        frames.append(df[df["model"] == model].assign(curve=label))
    return pd.concat(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curve", action="append", required=True, help="label=chemin.csv:modèle")
    parser.add_argument("--out", required=True, help="préfixe des sorties (.md, .png, .pdf)")
    parser.add_argument("--dashed", nargs="*", default=[], help="courbes tracées en tirets")
    args = parser.parse_args()

    df = load_curves(args.curve)
    labels = list(dict.fromkeys(df["curve"]))
    stats = (df.groupby(["curve", "degradation", "level"])["dice_fg_mean"]
             .agg(["mean", "std", "count"]).reset_index())

    # Tableau : une ligne par (dégradation, niveau), une colonne par courbe
    cell = stats.assign(v=stats["mean"].map("{:.3f}".format) + " ± " + stats["std"].fillna(0).map("{:.3f}".format))
    table = cell.pivot_table(index=["degradation", "level"], columns="curve", values="v", aggfunc="first")[labels]
    lines = ["| Dégradation | Niveau | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for (kind, level), row in table.iterrows():
        lines.append(f"| {kind} | {level} | " + " | ".join(row.values) + " |")
    md = "\n".join(lines)
    print(md)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md + "\n")

    # Figure
    clean = stats[stats["degradation"] == "none"].set_index("curve")["mean"]
    fig, axes = plt.subplots(1, len(KINDS), figsize=(11, 3.4), sharey=True)
    for ax, (kind, title) in zip(axes, KINDS.items()):
        for label, color in zip(labels, COLORS):
            s = stats[(stats["curve"] == label) & (stats["degradation"] == kind)].sort_values("level")
            style = "--" if label in args.dashed else "-"
            ax.errorbar(s["level"], s["mean"], yerr=s["std"].fillna(0), marker="o", markersize=4, capsize=3,
                        color=color, linestyle=style, label=label)
            ax.axhline(clean[label], color=color, linestyle=":", linewidth=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("degradation level (0 = weakest, 9 = strongest)", fontsize=8.5)
        ax.set_xticks(sorted(stats[stats["degradation"] == kind]["level"].unique()))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("artery Dice (test set)")
    axes[0].legend(fontsize=8, title="dotted: clean images", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=140)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"-> {out}.md, {out}.png, {out}.pdf")


if __name__ == "__main__":
    main()
