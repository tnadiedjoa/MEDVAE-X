"""Chargement des configs, surcharges --set et dossiers de runs."""

import json

import pytest
import yaml

from finetune import config as config_mod
from finetune import runs


def test_overrides_parse_values():
    cfg = runs.apply_overrides({"training": {"epochs": 100}}, [
        "training.learning_rate=1e-3",   # notation scientifique : float, pas une chaîne
        "training.epochs=5",
        "encoder.checkpoint_path=a/b.pth",
        "model.flag=true",
    ])
    assert cfg["training"] == {"epochs": 5, "learning_rate": 0.001}
    assert cfg["encoder"]["checkpoint_path"] == "a/b.pth"
    assert cfg["model"]["flag"] is True


def test_override_without_equal_sign_fails():
    with pytest.raises(ValueError):
        runs.apply_overrides({}, ["training.epochs"])


def test_load_config_resolves_data_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCADE_ROOT", str(tmp_path / "arcade"))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"data": {
        "train_images": "dataset_phase_1/imgs",
        "val_ann": "/absolu/ann.json",
        "num_classes": 26,
    }}))
    data = config_mod.load_config(str(path))["data"]
    assert data["train_images"] == str(tmp_path / "arcade" / "dataset_phase_1" / "imgs")
    assert data["val_ann"] == "/absolu/ann.json"   # un chemin absolu reste inchangé
    assert data["num_classes"] == 26


def test_create_run_writes_config_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("ARCADE_ROOT", "/data/arcade")
    cfg = {"experiment": {"name": "x"},
           "data": {"train_images": "/data/arcade/dataset_phase_1/imgs"}}
    run_dir = runs.create_run(cfg, "essai", ["training.epochs=1"])

    saved = yaml.safe_load(open(f"{run_dir}/config.yaml"))
    # Chemins du dataset sauvegardés relatifs à ARCADE_ROOT (portables)
    assert saved["data"]["train_images"] == "dataset_phase_1/imgs"
    assert cfg["data"]["train_images"] == "/data/arcade/dataset_phase_1/imgs"   # config d'origine intacte

    meta = json.load(open(f"{run_dir}/meta.json"))
    assert meta["run"].endswith("_essai")
    assert meta["overrides"] == ["training.epochs=1"]
    assert {"git_commit", "git_dirty", "gpu", "torch"} <= meta.keys()
