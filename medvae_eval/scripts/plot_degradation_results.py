import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv("../outputs/degradation/degradation_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Évolution des métriques par niveau de dégradation", fontsize=13, fontweight="bold")

for ax, col, label, color in [
    (axes[0], "psnr_mean",    "PSNR moyen (dB)",  "blue"),
    (axes[1], "ms_ssim_mean", "MS-SSIM moyen",     "orange"),
]:
    ax2 = ax.twinx()
    l1, = ax.plot(df["level"], df[col], color=color, marker="o", markersize=3, label=label)
    l2, = ax2.plot(df["level"], df["arniqa_mean"], color="gray", linestyle="--",
                   marker="s", markersize=2, alpha=0.7, label="ARNIQA moyen")
    ax.set_xlabel("Niveau de dégradation")
    ax.set_ylabel(label, color=color)
    ax2.set_ylabel("ARNIQA moyen", color="gray")
    ax.tick_params(axis="y", labelcolor=color)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.legend(handles=[l1, l2], loc="lower left")
    ax.grid(alpha=0.3)

    # annotation des paramètres clés
    for lvl, row in df.iterrows():
        if lvl % 10 == 0:
            ax.annotate(f"σ={int(row.noise_sigma)}\nk={int(row.blur_kernel)}\nq={int(row.jpeg_quality)}",
                        xy=(row.level, row[col]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=6, color="dimgray")

plt.tight_layout()
plt.savefig("../outputs/degradation/metrics_by_level.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PSNR / MS-SSIM vs ARNIQA (coloré par niveau)", fontsize=13, fontweight="bold")

for ax, col, label, cmap in [
    (axes[0], "psnr_mean",    "PSNR moyen (dB)", "Blues"),
    (axes[1], "ms_ssim_mean", "MS-SSIM moyen",    "Oranges"),
]:
    sc = ax.scatter(df["arniqa_mean"], df[col], c=df["level"],
                    cmap=cmap, s=30, zorder=3)
    ax.plot(df["arniqa_mean"], df[col], color="gray", alpha=0.3, linewidth=0.8, zorder=2)
    plt.colorbar(sc, ax=ax, label="Niveau de dégradation")
    ax.set_xlabel("ARNIQA moyen")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs ARNIQA")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../outputs/degradation/psnr_mssim_vs_arniqa_colored.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Rampe de dégradation", fontsize=12)
for ax, col, label, color in [
    (axes[0], "noise_sigma",  "Bruit gaussien σ",   "#e74c3c"),
    (axes[1], "blur_kernel",  "Noyau de flou (px)", "#3498db"),
    (axes[2], "jpeg_quality", "Qualité JPEG",        "#2ecc71"),
]:
    ax.plot(df["level"], df[col], color=color, marker="o", markersize=3)
    ax.set_xlabel("Niveau")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../outputs/degradation/degradation_ramp.png", dpi=150, bbox_inches="tight")
plt.show()