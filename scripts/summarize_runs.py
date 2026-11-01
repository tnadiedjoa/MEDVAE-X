"""Summarise groups of runs (several seeds): mean ± standard deviation of the scores.

The val_dice column is the best validation Dice (seg_val, with the background): decisions
are made on it; the other columns are test scores.

Each group is "name=pattern", the pattern being a glob on the folder names of
experiments/runs/ (several patterns separated by commas). With --ref, also prints the
gap of each group to the reference group, seed by seed (runs paired by seed).

Example (from the repository root):
    python scripts/summarize_runs.py --ref A \\
        "A=*_e02_condition_a,*_e06_seed4?_condition_a" "E08=*_e08_official_val_seed*"
"""

import argparse
import fnmatch
import json
import statistics
from pathlib import Path

import yaml

RUNS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "runs"
METRICS = ("val_dice", "dice_mean", "dice_fg_mean", "iou_fg_mean")


def best_val_dice(run_dir: Path):
    """Best validation Dice of the run history (None if there is none)."""
    history = next(iter(run_dir.glob("*_history.json")), None)
    if history is None:
        return None
    val = json.loads(history.read_text()).get("val", [])
    return max((e["dice_mean"] for e in val), default=None)


def load_group(patterns: str) -> dict:
    """{seed: scores} for the runs whose name matches one of the patterns."""
    runs = {}
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not any(fnmatch.fnmatch(run_dir.name, p) for p in patterns.split(",")):
            continue
        results = run_dir / "results.json"
        if not results.exists():
            continue
        (_, scores), = json.loads(results.read_text()).items()
        seed = yaml.safe_load((run_dir / "config.yaml").read_text())["experiment"]["seed"]
        if seed in runs:
            raise SystemExit(f"Two runs with seed {seed} for \"{patterns}\": {run_dir.name}")
        runs[seed] = {**scores, "run": run_dir.name}
        val = best_val_dice(run_dir)
        if val is not None:
            runs[seed]["val_dice"] = val
    return runs


def fmt(values: list) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", help="nom=motif[,motif...]")
    parser.add_argument("--ref", help="name of the reference group for the paired gaps")
    args = parser.parse_args()

    groups = {}
    for spec in args.groups:
        name, _, patterns = spec.partition("=")
        groups[name] = load_group(patterns)

    print(f"| Groupe | Seeds | {' | '.join(METRICS)} |")
    print("|---|---|" + "---|" * len(METRICS))
    for name, runs in groups.items():
        cells = [fmt([r[m] for r in runs.values() if m in r]) if any(m in r for r in runs.values()) else "—"
                 for m in METRICS]
        print(f"| {name} | {', '.join(map(str, sorted(runs)))} | {' | '.join(cells)} |")

    if args.ref:
        ref = groups[args.ref]
        for metric in ("val_dice", "dice_mean"):
            print(f"\nGap in {metric} to group {args.ref}, seed by seed:")
            for name, runs in groups.items():
                if name == args.ref:
                    continue
                common = [s for s in sorted(set(runs) & set(ref)) if metric in runs[s] and metric in ref[s]]
                diffs = [runs[s][metric] - ref[s][metric] for s in common]
                if diffs:
                    print(f"  {name:12s} " + ", ".join(f"{d:+.4f}" for d in diffs)
                          + f"   (moyenne {statistics.mean(diffs):+.4f})")


if __name__ == "__main__":
    main()
