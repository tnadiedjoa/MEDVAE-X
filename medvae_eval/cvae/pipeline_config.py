"""
Shared configuration of the inductive-bias pipeline.
Change APPROACH to change how the quality score c is computed.

Notebooks to rerun after a change of approach:
  - Approach A (weighted score, weights = PCA loadings) or B (PCA PC1): rerun NB3 → NB4 → NB5.
  - Approach C (learned MLP score + ResNet-18): rerun NB2 → NB3 → NB4 → NB5
    (NB2 must be rerun or available, since approach C needs the ResNet-18 features).
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN HYPERPARAMETER — how the score c is computed           ║
# ╚══════════════════════════════════════════════════════════════╝

APPROACH = "C"

# ── Approaches ────────────────────────────────────────────────

APPROACHES = {
    "A": {
        "score_col": "score_weighted",
        "name": "Weighted score (PCA weights)",
        "use_resnet": False,
        "description": (
            "Linear combination of 6 min-max normalised IQA metrics, "
            "weighted by the absolute PC1 loadings (|loading| / Σ|loading|, "
            "fitted on TRAIN). Objective and reproducible weights, no more "
            "arbitrary manual values."
        ),
    },
    "B": {
        "score_col": "score_pca",
        "name": "PCA (PC1)",
        "use_resnet": False,
        "description": (
            "First principal component normalised to [0,1], sign corrected "
            "through the correlation with Tenengrad. Objective, data-driven."
        ),
    },
    "C": {
        "score_col": "score_dl",
        "name": "Learned score (MLP + ResNet-18)",
        "use_resnet": True,
        "description": (
            "MLP (512→128→32→1) trained with MSE on the frozen ResNet-18 features "
            "of synthetically degraded TRAIN images (noise/blur/JPEG), "
            "with a quality target = 1 - known severity. The score is then "
            "applied to the real ResNet-18 features (NB2). Supervision = synthetic "
            "proxy (ARCADE has no real quality annotation)."
        ),
    },
}


def get_approach_config(approach=None):
    """Return the config of the selected approach."""
    approach = approach or APPROACH
    if approach not in APPROACHES:
        raise ValueError(
            f"Unknown approach '{approach}'. Choices: {list(APPROACHES.keys())}"
        )
    return APPROACHES[approach]


def get_score_column(approach=None):
    """Return the name of the score column of the approach."""
    return get_approach_config(approach)["score_col"]


def use_resnet(approach=None):
    """Return True if the approach needs the ResNet-18 features."""
    return get_approach_config(approach)["use_resnet"]


def print_approach_summary(approach=None):
    """Print a summary of the selected approach."""
    approach = approach or APPROACH
    cfg = get_approach_config(approach)
    print(f"╔{'═' * 58}╗")
    print(f"║  Approach: {approach} — {cfg['name']:<44s} ║")
    print(f"╚{'═' * 58}╝")
    print(f"  Colonne de score : {cfg['score_col']}")
    print(f"  {cfg['description']}")
    print()


# ── Chemins ───────────────────────────────────────────────────

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Root of the ARCADE dataset: $ARCADE_ROOT if set, otherwise <repo>/data/arcade
ARCADE_ROOT = Path(os.environ.get("ARCADE_ROOT", _REPO_ROOT / "data" / "arcade"))


def resolve_image_path(path) -> str:
    """Image path valid on this machine.

    The CSV files (NB2, NB3) store absolute paths specific to the machine that
    produced them (e.g. /home/infres/<other>/arcade_challenge_datasets/...): keep the
    part relative to the ARCADE root and attach it to ARCADE_ROOT.
    """
    path = str(path)
    for marker in ("arcade_challenge_datasets/", "/dataset_phase_1/", "/dataset_final_phase/"):
        if marker in path:
            relative = path.split(marker, 1)[1]
            if marker.startswith("/"):
                relative = marker.strip("/") + "/" + relative
            return str(ARCADE_ROOT / relative)
    return path


def read_csv_with_paths(csv_path):
    """pd.read_csv, with the "path" column converted by resolve_image_path."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "path" in df.columns:
        df["path"] = df["path"].map(resolve_image_path)
    return df
