import os
import pandas as pd
import numpy as np
import scipy.stats as stats

# Sorties dans <repo>/medvae_eval/outputs/, quel que soit le dossier courant
_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

arniqa = pd.read_csv(f"{_OUTPUTS}/metrics/arniqa_scores.csv")
fr = pd.read_csv(f"{_OUTPUTS}/metrics/FR_results.csv")

df = pd.DataFrame({
    'arniqa': arniqa['arniqa_score'].values,
    'psnr': fr['psnr'].values,
}).replace([np.inf, -np.inf], np.nan).dropna()

a, p = df['arniqa'], df['psnr']

print("--- ARNIQA (dataset sain, sans dégradation) ---")
print(f"n          : {len(df)}")
print(f"min / max  : {a.min():.4f} / {a.max():.4f}")
print(f"range      : {a.max() - a.min():.4f}")
print(f"mean ± std : {a.mean():.4f} ± {a.std():.4f}")
print(f"quantiles 5/50/95 : {a.quantile(0.05):.4f} / {a.median():.4f} / {a.quantile(0.95):.4f}\n")

pearson_corr, pearson_pval = stats.pearsonr(a, p)
spearman_corr, spearman_pval = stats.spearmanr(a, p)
print("--- ARNIQA vs PSNR de reconstruction ---")
print(f"Pearson:  {pearson_corr:.4f} (p={pearson_pval:.4e})")
print(f"Spearman: {spearman_corr:.4f} (p={spearman_pval:.4e})")