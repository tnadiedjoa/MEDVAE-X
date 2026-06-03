import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "../outputs/degradation"
METRIC = "arniqa"
ENGINEERED_METHOD = "weighted"

metric_label = f"{METRIC}_{ENGINEERED_METHOD}" if METRIC == "engineered" else METRIC
score_col = f"{metric_label}_mean"
score_label = "ARNIQA moyen" if METRIC == "arniqa" else f"Engineered Score ({ENGINEERED_METHOD}) moyen"

df = pd.read_csv(f"{OUTPUT_DIR}/degradation_results_{metric_label}.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Évolution des métriques par niveau de dégradation ({score_label})", fontsize=13, fontweight="bold")
for ax, col, label, color in [
    (axes[0], "psnr_mean",    "PSNR moyen (dB)", "blue"),
    (axes[1], "ms_ssim_mean", "MS-SSIM moyen",   "orange"),
]:
    ax2 = ax.twinx()
    l1, = ax.plot(df["level"], df[col], color=color, marker="o", markersize=3, label=label)
    l2, = ax2.plot(df["level"], df[score_col], color="gray", linestyle="--",
                   marker="s", markersize=2, alpha=0.7, label=score_label)
    ax.set_xlabel("Niveau de dégradation")
    ax.set_ylabel(label, color=color)
    ax2.set_ylabel(score_label, color="gray")
    ax.tick_params(axis="y", labelcolor=color)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.legend(handles=[l1, l2], loc="lower left")
    ax.grid(alpha=0.3)
    for lvl, row in df.iterrows():
        if lvl % 10 == 0:
            ax.annotate(f"σ={int(row.noise_sigma)}\nk={int(row.blur_kernel)}\nq={int(row.jpeg_quality)}",
                        xy=(row.level, row[col]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=6, color="dimgray")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_by_level_{metric_label}.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"PSNR / MS-SSIM vs {score_label} (coloré par niveau)", fontsize=13, fontweight="bold")
for ax, col, label, cmap in [
    (axes[0], "psnr_mean",    "PSNR moyen (dB)", "Blues"),
    (axes[1], "ms_ssim_mean", "MS-SSIM moyen",   "Oranges"),
]:
    sc = ax.scatter(df[score_col], df[col], c=df["level"], cmap=cmap, s=30, zorder=3)
    ax.plot(df[score_col], df[col], color="gray", alpha=0.3, linewidth=0.8, zorder=2)
    plt.colorbar(sc, ax=ax, label="Niveau de dégradation")
    ax.set_xlabel(score_label)
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs {score_label}")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/psnr_mssim_vs_{metric_label}_colored.png", dpi=150, bbox_inches="tight")
plt.show()