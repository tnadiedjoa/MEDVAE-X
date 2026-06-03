import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRIC = "arniqa"
ENGINEERED_METHOD = "weighted"

metric_label = f"{METRIC}_{ENGINEERED_METHOD}" if METRIC == "engineered" else METRIC
score_col = f"{metric_label}_score"
score_label = "ARNIQA" if METRIC == "arniqa" else f"Engineered ({ENGINEERED_METHOD})"


def plot_metrics(metrics_dict, save_path):
    quality_scores = metrics_dict["score"]
    psnr_scores = metrics_dict["psnr"]
    ms_ssim_scores = metrics_dict["ms_ssim"]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(quality_scores, psnr_scores, color='blue', label='PSNR')
    plt.xlabel(f'{score_label} Score')
    plt.ylabel('PSNR')
    plt.title(f'PSNR vs {score_label} Score')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.scatter(quality_scores, ms_ssim_scores, color='orange', label='MS-SSIM')
    plt.xlabel(f'{score_label} Score')
    plt.ylabel('MS-SSIM')
    plt.title(f'MS-SSIM vs {score_label} Score')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_correlation_matrix(metrics_df, save_path):
    corr_data = metrics_df[['psnr', 'ms_ssim', score_col]].copy()
    corr_data.columns = ['PSNR', 'MS-SSIM', score_label]
    correlation = corr_data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=2, cbar_kws={'label': 'Correlation'},
                vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix: PSNR, MS-SSIM, {score_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print("\nCorrelation Matrix:")
    print(correlation)


if __name__ == "__main__":
    recons_df = pd.read_csv("../outputs/metrics/FR_results.csv")
    score_df = pd.read_csv(f"../outputs/metrics/{metric_label}_scores.csv")
    recons_df["image"] = recons_df["image"].astype(float)
    score_df["image"] = score_df["image"].astype(float)
    merged_df = pd.merge(recons_df, score_df, on="image")
    metrics_dict = {
        "score": merged_df[score_col].tolist(),
        "psnr": merged_df["psnr"].tolist(),
        "ms_ssim": merged_df["ms_ssim"].tolist(),
    }
    plot_metrics(metrics_dict, f"../outputs/metrics/plot_metrics_{metric_label}.png")
    plot_correlation_matrix(merged_df, f"../outputs/metrics/correlation_matrix_{metric_label}.png")