"""Un dossier par run : config utilisée, métadonnées (commit, GPU, job) et sorties.

Chaque run est rangé dans experiments/runs/<date>_<nom>/. Les petits fichiers
(config, historique, scores) sont versionnés ; les checkpoints .pth sont ignorés
par git.
"""

import json
import os
import socket
import subprocess
from datetime import datetime

import torch
import yaml

from finetune.config import DATA_PATH_KEYS, REPO_ROOT, arcade_root

RUNS_DIR = REPO_ROOT / "experiments" / "runs"


def add_run_args(parser) -> None:
    """Options communes aux scripts qui lancent un run."""
    parser.add_argument("--run-name", type=str, default=None,
                        help="Nom du run (défaut : experiment.name de la config)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="CLE=VALEUR",
                        help="Surcharge une valeur de la config, ex. training.epochs=1 "
                             "(répétable)")


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Applique des surcharges 'a.b.c=valeur' ; la valeur est lue comme du YAML."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Surcharge invalide (attendu CLE=VALEUR) : {item}")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        value = yaml.safe_load(raw)
        # PyYAML lit « 1e-3 » comme du texte (il faut « 1.0e-3 » en YAML 1.1)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        node[leaf] = value
    return config


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def create_run(config: dict, name: str, overrides: list[str] = ()) -> str:
    """Crée le dossier du run, y écrit config.yaml et meta.json, et renvoie son chemin."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Config sauvegardée avec les chemins du dataset relatifs à ARCADE_ROOT,
    # comme dans les yaml d'origine (pas de chemin propre à la machine)
    saved = {**config, "data": dict(config.get("data", {}))}
    for key in DATA_PATH_KEYS:
        if key in saved["data"]:
            path = saved["data"][key]
            root = str(arcade_root())
            if path.startswith(root + os.sep):
                saved["data"][key] = os.path.relpath(path, root)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(saved, f, sort_keys=False, allow_unicode=True)

    meta = {
        "run": run_dir.name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        # Modifications non commitées au moment du lancement (le run n'est alors
        # pas exactement reproductible depuis le commit)
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "overrides": list(overrides),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dossier du run : {run_dir}")
    return str(run_dir)
