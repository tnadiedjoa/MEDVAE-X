import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_metrics(metrics_dict, save_path):
    # plot PSNR and MS-SSIM with respect to arniqa scores
    arniqa_scores = metrics_dict["arniqa"]
    psnr_scores = metrics_dict["psnr"]
    ms_ssim_scores = metrics_dict["ms_ssim"]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(arniqa_scores, psnr_scores, color='blue', label='PSNR')
    plt.xlabel('Arniqa Score')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Arniqa Score')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(arniqa_scores, ms_ssim_scores, color='orange', label='MS-SSIM')
    plt.xlabel('Arniqa Score')
    plt.ylabel('MS-SSIM')
    plt.title('MS-SSIM vs Arniqa Score')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_correlation_matrix(metrics_df, save_path):
    # create correlation matrix between PSNR, MS-SSIM, and ARNIQA
    corr_data = metrics_df[['psnr', 'ms_ssim', 'arniqa_score']].copy()
    corr_data.columns = ['PSNR', 'MS-SSIM', 'ARNIQA']
    
    correlation = corr_data.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                square=True, linewidths=2, cbar_kws={'label': 'Correlation'}, 
                vmin=-1, vmax=1)
    plt.title('Correlation Matrix: PSNR, MS-SSIM, ARNIQA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nCorrelation Matrix:")
    print(correlation)

if __name__ == "__main__":
    recons_df = pd.read_csv("../outputs/metrics/reconstruction_metrics.csv")
    arniqa_df = pd.read_csv("../outputs/metrics/arniqa_scores.csv")
    merged_df = pd.merge(recons_df, arniqa_df, on="image")
    metrics_dict = {
        "arniqa": merged_df["arniqa_score"].tolist(),
        "psnr": merged_df["psnr"].tolist(),
        "ms_ssim": merged_df["ms_ssim"].tolist()
    }
    plot_metrics(metrics_dict, "../outputs/metrics/plot_metrics.png")
    plot_correlation_matrix(merged_df, "../outputs/metrics/correlation_matrix.png")
