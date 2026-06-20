import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "../outputs/degradation"
MODE = "nr"  # "nr" (arniqa) ou "masked" (poisson/jpeg/blur)
DEGRADATION = "poisson"  # "poisson" ou "jpeg" ou "blur" (si MODE="masked")

if MODE == "nr":
    metric_label = "arniqa"
    score_col = "masked_psnr_degra"
    cleanrec_col = "masked_psnr_cleanrec"
    score_label = "ARNIQA moyen (image dégradée)"
    csv_path = f"{OUTPUT_DIR}/arniqa_sweep_{DEGRADATION}.csv"
    recon_cols = [("masked_psnr_recon", "MaskedPSNR reconstruction (dB)", "blue", "Blues")]
    suffix = metric_label
    annot_cols = ["param"]
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
clean_cols = [c for c, *_ in recon_cols] + [score_col] + ([cleanrec_col] if MODE == "masked" else [])
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=clean_cols)

all_y_values = [df[col].replace([np.inf, -np.inf], np.nan).dropna().values for col, _, _, _ in recon_cols]
all_y_values.append(df[score_col].replace([np.inf, -np.inf], np.nan).dropna().values)
if cleanrec_col in df.columns:
    all_y_values.append(df[cleanrec_col].replace([np.inf, -np.inf], np.nan).dropna().values)
y_min = min([arr.min() for arr in all_y_values])
y_max = max([arr.max() for arr in all_y_values])
y_padding = (y_max - y_min) * 0.05

fig, axes = plt.subplots(1, len(recon_cols), figsize=(7 * len(recon_cols), 5), squeeze=False)
axes = axes[0]
fig.suptitle(f"Évolution des métriques par niveau de dégradation ({score_label})", fontsize=13, fontweight="bold")
for ax, (col, label, color, _) in zip(axes, recon_cols):
    ax2 = ax.twinx()
    l1, = ax.plot(df["level"], df[col], color=color, marker="o", markersize=3, label=label)
    l2, = ax2.plot(df["level"], df[score_col], color="gray", linestyle="--",
                   marker="s", markersize=2, alpha=0.7, label=score_label)
    handles = [l1, l2]
    if cleanrec_col in df.columns:
        l3, = ax2.plot(df["level"], df[cleanrec_col], color="green", linestyle=":",
                       marker="^", markersize=2, alpha=0.8,
                       label="MaskedPSNR clean vs recon (dB)")
        handles.append(l3)
    if MODE == "masked":
        cols = [col, score_col] + ([cleanrec_col] if cleanrec_col in df.columns else [])
        lo, hi = df[cols].min().min(), df[cols].max().max()
        pad = (hi - lo) * 0.05
        ax.set_ylim(lo - pad, hi + pad)
        ax2.set_ylim(lo - pad, hi + pad)
    else:
        left_cols = [col] + ([cleanrec_col] if cleanrec_col in df.columns else [])
        lo, hi = df[left_cols].min().min(), df[left_cols].max().max()
        pad = (hi - lo) * 0.05
        ax.set_ylim(lo - pad, hi + pad)
        slo, shi = df[score_col].min(), df[score_col].max()
        spad = (shi - slo) * 0.05 or 0.01
        ax2.set_ylim(slo - spad, shi + spad)
    ax.set_xlabel("Niveau de dégradation")
    ax.set_ylabel(label, color=color)
    ax2.set_ylabel(score_label, color="gray")
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax2.set_ylim(y_min - y_padding, y_max + y_padding)
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
