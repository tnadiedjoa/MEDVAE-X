import sys
from pathlib import Path

# Imports « finetune.* » et « scripts/ » depuis la racine du repo
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
