"""Promotion d'un run et résolution des chemins de l'axe B."""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promote_run_merges_scores_and_copies_history(tmp_path, monkeypatch):
    promote = load_module("promote_run", REPO_ROOT / "scripts" / "promote_run.py")
    results = tmp_path / "results"
    results.mkdir()
    (results / "results.json").write_text(json.dumps({"condition_a": {"dice_mean": 0.4}}))
    monkeypatch.setattr(promote, "RESULTS_DIR", results)
    monkeypatch.setattr(promote, "REPO_ROOT", tmp_path)

    run = tmp_path / "2026-01-01_000000_essai"
    run.mkdir()
    (run / "meta.json").write_text("{}")
    (run / "condition_b_medvae_history.json").write_text("{}")
    (run / "results.json").write_text(json.dumps({"condition_b": {"dice_mean": 0.41}}))
    promote.promote(run)

    official = json.loads((results / "results.json").read_text())
    assert official["condition_a"] == {"dice_mean": 0.4}                  # autres conditions intactes
    assert official["condition_b"] == {"dice_mean": 0.41, "run": run.name}
    assert (results / "condition_b_medvae_history.json").exists()


def test_resolve_image_path_rebases_foreign_paths(monkeypatch):
    monkeypatch.setenv("ARCADE_ROOT", "/data/arcade")
    sys.path.insert(0, str(REPO_ROOT / "medvae_eval" / "cvae"))
    pipeline_config = load_module("pipeline_config", REPO_ROOT / "medvae_eval" / "cvae" / "pipeline_config.py")
    resolve = pipeline_config.resolve_image_path
    foreign = "/home/infres/autre/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train/images/1.png"
    assert resolve(foreign) == "/data/arcade/dataset_phase_1/segmentation_dataset/seg_train/images/1.png"
    assert resolve("/x/data/arcade/dataset_final_phase/test/images/2.png") == "/data/arcade/dataset_final_phase/test/images/2.png"
    assert resolve("autre/chose.png") == "autre/chose.png"
