"""Make a validated run the official reference of the project (finetune/results/).

Copies the history of the run and merges its scores into finetune/results/results.json,
recording the source run. To run once the experiment is logged in experiments.md and
the change kept.

Usage (from the repository root):
    python scripts/promote_run.py experiments/runs/<run> [experiments/runs/<run> ...]
"""

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "finetune" / "results"


def promote(run_dir: Path) -> None:
    if not (run_dir / "meta.json").exists():
        raise SystemExit(f"{run_dir} is not a run folder (no meta.json)")

    # Training histories (the MedVAE fine-tuning has its own subfolder)
    for history in run_dir.glob("*_history.json"):
        dest = RESULTS_DIR / ("medvae_finetuned" if history.name.startswith("medvae_") else "")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history, dest / history.name)
        print(f"  {history.name} -> {dest.relative_to(REPO_ROOT)}/")

    # Test scores: one entry per condition, with the run that produced it
    run_results = run_dir / "results.json"
    if run_results.exists():
        official_path = RESULTS_DIR / "results.json"
        official = json.loads(official_path.read_text()) if official_path.exists() else {}
        for condition, scores in json.loads(run_results.read_text()).items():
            official[condition] = {**scores, "run": run_dir.name}
            print(f"  {condition} -> results.json")
        official_path.write_text(json.dumps(official, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    for run_dir in parser.parse_args().runs:
        print(f"Promotion de {run_dir.name}")
        promote(run_dir.resolve())


if __name__ == "__main__":
    main()
