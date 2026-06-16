import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "../outputs/degradation"
MODE = "masked"  # "nr" (arniqa/engineered) ou "masked" (poisson/jpeg)
METRIC = "arniqa"
ENGINEERED_METHOD = "weighted"
DEGRADATION = "poisson"  # "poisson" ou "jpeg" ou "blur" (si MODE="masked")

if MODE == "nr":
    metric_label = f"{METRIC}_{ENGINEERED_METHOD}" if METRIC == "engineered" else METRIC
    score_col = f"{metric_label}_mean"
    score_label = "ARNIQA moyen" if METRIC == "arniqa" else f"Engineered Score ({ENGINEERED_METHOD}) moyen"
    csv_path = f"{OUTPUT_DIR}/degradation_results_{metric_label}.csv"
    recon_cols = [("psnr_mean", "PSNR moyen (dB)", "blue", "Blues"),
                  ("ms_ssim_mean", "MS-SSIM moyen", "orange", "Oranges")]
    suffix = metric_label
    annot_cols = ["noise_sigma", "blur_kernel", "jpeg_quality"]
else:
    metric_label = f"masked_{DEGRADATION}"
    score_col = "masked_psnr_degra"
    cleanrec_col = "masked_psnr_cleanrec" 
    score_label = "MaskedPSNR dégradée vs clean (dB)"
    csv_path = f"{OUTPUT_DIR}/masked_sweep_{DEGRADATION}.csv"
    recon_cols = [("masked_psnr_recon", "MaskedPSNR reconstruction (dB)", "blue", "Blues")]
    suffix = metric_label
    annot_cols = ["param"]

df = pd.read_csv(csv_path)

fig, axes = plt.subplots(1, len(recon_cols), figsize=(7 * len(recon_cols), 5), squeeze=False)
axes = axes[0]
fig.suptitle(f"Évolution des métriques par niveau de dégradation ({score_label})", fontsize=13, fontweight="bold")
for ax, (col, label, color, _) in zip(axes, recon_cols):
    ax2 = ax.twinx()
    l1, = ax.plot(df["level"], df[col], color=color, marker="o", markersize=3, label=label)
    l2, = ax2.plot(df["level"], df[score_col], color="gray", linestyle="--",
                   marker="s", markersize=2, alpha=0.7, label=score_label)
    handles = [l1, l2]
    if MODE == "masked" and cleanrec_col in df.columns:
        l3, = ax2.plot(df["level"], df[cleanrec_col], color="green", linestyle=":",
                        marker="^", markersize=2, alpha=0.8,
                        label="MaskedPSNR clean vs recon (dB)")
        handles.append(l3)
    ax.set_xlabel("Niveau de dégradation")
    ax.set_ylabel(label, color=color)
    ax2.set_ylabel(score_label, color="gray")
    ax.tick_params(axis="y", labelcolor=color)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.legend(handles=handles, loc="lower left")
    ax.grid(alpha=0.3)
    for lvl, row in df.iterrows():
        if lvl % 10 == 0:
            txt = "\n".join(f"{c}={row[c]:g}" for c in annot_cols)
            ax.annotate(txt, xy=(row.level, row[col]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=6, color="dimgray")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_by_level_{suffix}.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, len(recon_cols), figsize=(7 * len(recon_cols), 5), squeeze=False)
axes = axes[0]
fig.suptitle(f"Reconstruction vs {score_label} (coloré par niveau)", fontsize=13, fontweight="bold")
for ax, (col, label, _, cmap) in zip(axes, recon_cols):
    sc = ax.scatter(df[score_col], df[col], c=df["level"], cmap=cmap, s=30, zorder=3)
    ax.plot(df[score_col], df[col], color="gray", alpha=0.3, linewidth=0.8, zorder=2)
    plt.colorbar(sc, ax=ax, label="Niveau de dégradation")
    ax.set_xlabel(score_label)
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs {score_label}")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/recon_vs_score_{suffix}_colored.png", dpi=150, bbox_inches="tight")
plt.show()