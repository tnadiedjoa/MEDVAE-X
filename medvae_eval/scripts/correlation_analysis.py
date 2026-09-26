import os
import pandas as pd
import numpy as np
import scipy.stats as stats

# Sorties dans <repo>/medvae_eval/outputs/, quel que soit le dossier courant
_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

files = [f"{_OUTPUTS}/degradation/masked_sweep_blur.csv", f"{_OUTPUTS}/degradation/masked_sweep_jpeg.csv", f"{_OUTPUTS}/degradation/masked_sweep_poisson.csv"]

for file in files:
    df = pd.read_csv(file)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['masked_psnr_recon', 'masked_psnr_degra'])
    
    recon = df['masked_psnr_recon']
    degra = df['masked_psnr_degra']
    
    pearson_corr, pearson_pval = stats.pearsonr(recon, degra)
    spearman_corr, spearman_pval = stats.spearmanr(recon, degra)
    
    print(f"--- {file} ---")
    print(f"Pearson: {pearson_corr:.4f} (p-value: {pearson_pval:.4e})")
    print(f"Spearman: {spearman_corr:.4f} (p-value: {spearman_pval:.4e})\n")