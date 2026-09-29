"""Résumé des runs FiLM (train_film.py) : niveaux et différences appariées.

Pour chaque groupe (motif de dossiers de runs), moyenne ± écart-type entre seeds des
métriques de test. Puis deux types de différences appariées :
    - dans un même run FiLM : vrai c − c mélangé, vrai c − c constant (si c est utilisé,
      ces différences doivent être nettement positives). Même modèle : la seule source
      de variation est l'image, d'où un IC 95 % bootstrap sur les images ;
    - entre groupes, à seed égal : groupe − référence (variante real_c). Ce sont des
      modèles entraînés séparément : l'IC 95 % porte sur les seeds (loi de Student sur
      les écarts moyens par seed), l'IC sur les images ignorerait la variabilité
      d'entraînement. Avec un seul seed, pas d'IC.

Usage (racine du repo) :
    python medvae_eval/film/analyze.py \
        --group pretrained='*_e13_film_pretrained' \
        --group baseline='*_e13_film_baseline_seed4?' \
        --group film_C='*_e13_film_C_seed4?' \
        --ref baseline --out experiments/film/e13
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from finetune.runs import RUNS_DIR  # noqa: E402

METRICS = ("psnr", "mpsnr", "ssim", "haarpsi")


def load_group(pattern: str) -> dict:
    """{seed: tableau par image} pour tous les runs correspondant au motif."""
    runs = {}
    for run in sorted(RUNS_DIR.glob(pattern)):
        csv = run / "test_per_image.csv"
        if not csv.exists():
            print(f"  (ignoré, pas de test_per_image.csv) {run.name}")
            continue
        m = re.search(r"seed(\d+)", run.name)
        runs[int(m.group(1)) if m else 0] = pd.read_csv(csv)
    return runs


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(diffs_per_seed: list, metric: str) -> dict:
    """diffs_per_seed : Series indexées par image. Moyenne par seed, puis IC sur les images."""
    per_seed = [float(d.mean()) for d in diffs_per_seed]
    per_image = pd.concat(diffs_per_seed, axis=1).mean(axis=1).to_numpy()
    lo, hi = bootstrap_ci(per_image)
    return {"metric": metric, "mean": float(per_image.mean()), "ci_low": lo, "ci_high": hi, "ci": "images",
            "per_seed": " / ".join(f"{v:+.4f}" for v in per_seed)}


def matched_pairs(runs: dict, ref: dict) -> list:
    """Runs appariés par seed ; une référence à un seul run (ex. pré-entraîné) sert pour tous."""
    common = sorted(set(runs) & set(ref))
    if common:
        return [(runs[s], ref[s]) for s in common]
    if len(ref) == 1:
        only = next(iter(ref.values()))
        return [(df, only) for df in runs.values()]
    return []


def paired_seeds(diffs_per_seed: list, metric: str) -> dict:
    """Écarts entre modèles entraînés séparément : IC de Student sur les moyennes par seed."""
    from scipy import stats
    per_seed = np.array([float(d.mean()) for d in diffs_per_seed])
    mean, n = float(per_seed.mean()), len(per_seed)
    if n > 1:
        half = stats.t.ppf(0.975, n - 1) * per_seed.std(ddof=1) / np.sqrt(n)
        lo, hi = mean - half, mean + half
    else:
        lo = hi = float("nan")
    return {"metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "ci": "seeds",
            "per_seed": " / ".join(f"{v:+.4f}" for v in per_seed)}


def variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["variant"] == name].set_index("image")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True, help="nom=motif (glob dans experiments/runs)")
    parser.add_argument("--ref", default=None, help="groupe de référence pour les différences entre groupes")
    parser.add_argument("--out", default=None, help="dossier où écrire levels.csv et paired.csv")
    args = parser.parse_args()

    groups = {}
    for g in args.group:
        name, pattern = g.split("=", 1)
        groups[name] = load_group(pattern)
        print(f"{name} : {len(groups[name])} run(s), seeds {sorted(groups[name])}")

    # Niveaux : moyenne ± écart-type entre seeds
    levels = []
    for name, runs in groups.items():
        for v in sorted({v for df in runs.values() for v in df["variant"].unique()}):
            means = pd.DataFrame([variant(df, v)[list(METRICS)].mean() for df in runs.values()])
            row = {"group": name, "variant": v, "n_seeds": len(means)}
            for m in METRICS:
                row[m] = means[m].mean()
                row[f"{m}_std"] = means[m].std(ddof=1) if len(means) > 1 else 0.0
            levels.append(row)
    levels = pd.DataFrame(levels)
    print("\n| groupe | variante | seeds | " + " | ".join(METRICS) + " |")
    print("|---|---|---|" + "---|" * len(METRICS))
    for _, r in levels.iterrows():
        cells = [f"{r[m]:.4f} ± {r[f'{m}_std']:.4f}" if r["n_seeds"] > 1 else f"{r[m]:.4f}" for m in METRICS]
        print(f"| {r['group']} | {r['variant']} | {r['n_seeds']} | " + " | ".join(cells) + " |")

    # Différences appariées
    rows = []
    for name, runs in groups.items():
        for control in ("shuffled_c", "constant_c"):
            if not all(control in df["variant"].values for df in runs.values()):
                continue
            for m in METRICS:
                diffs = [variant(df, "real_c")[m] - variant(df, control)[m] for df in runs.values()]
                rows.append({"comparison": f"{name} : real_c − {control}", **paired(diffs, m)})
        if args.ref and name != args.ref:
            pairs = matched_pairs(runs, groups[args.ref])
            for m in METRICS:
                if pairs:
                    diffs = [variant(a, "real_c")[m] - variant(b, "real_c")[m] for a, b in pairs]
                    rows.append({"comparison": f"{name} − {args.ref}", **paired_seeds(diffs, m)})
    paired_df = pd.DataFrame(rows)
    print("\n| comparaison | métrique | différence moyenne | IC 95 % | IC sur | par seed |")
    print("|---|---|---|---|---|---|")
    for _, r in paired_df.iterrows():
        ci = "—" if np.isnan(r["ci_low"]) else f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"| {r['comparison']} | {r['metric']} | {r['mean']:+.4f} | {ci} | {r['ci']} | {r['per_seed']} |")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        levels.to_csv(out / "levels.csv", index=False)
        paired_df.to_csv(out / "paired.csv", index=False)
        print(f"\n-> {out}/levels.csv, {out}/paired.csv")


if __name__ == "__main__":
    main()
