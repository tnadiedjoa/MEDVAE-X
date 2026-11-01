"""Loading of the YAML configs and resolution of the ARCADE dataset paths."""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keys of the `data` section expressed relative to the ARCADE root
DATA_PATH_KEYS = ("train_images", "train_ann", "val_images", "val_ann",
                  "official_val_images", "official_val_ann")


def arcade_root() -> Path:
    """Root of the ARCADE dataset: $ARCADE_ROOT if set, otherwise <repo>/data/arcade."""
    return Path(os.environ.get("ARCADE_ROOT", REPO_ROOT / "data" / "arcade"))


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # An absolute path in the yaml stays unchanged (Path / absolute → absolute)
    data_cfg = config.get("data", {})
    for key in DATA_PATH_KEYS:
        if key in data_cfg:
            data_cfg[key] = str(arcade_root() / data_cfg[key])
    return config
