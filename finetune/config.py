"""Chargement des configs YAML et résolution des chemins du dataset ARCADE."""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Clés de la section `data` exprimées relativement à la racine ARCADE
DATA_PATH_KEYS = ("train_images", "train_ann", "val_images", "val_ann")


def arcade_root() -> Path:
    """Racine du dataset ARCADE : $ARCADE_ROOT si défini, sinon <repo>/data/arcade."""
    return Path(os.environ.get("ARCADE_ROOT", REPO_ROOT / "data" / "arcade"))


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Un chemin absolu dans le yaml reste inchangé (Path / absolu → absolu)
    data_cfg = config.get("data", {})
    for key in DATA_PATH_KEYS:
        if key in data_cfg:
            data_cfg[key] = str(arcade_root() / data_cfg[key])
    return config
