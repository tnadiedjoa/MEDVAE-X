import sys
from pathlib import Path

# Imports of "finetune.*" and "scripts/" from the repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
