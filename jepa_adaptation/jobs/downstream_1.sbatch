#!/bin/bash
#SBATCH -p H100,A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=jepa_adaptation/jobs/logs/downstream_1_%j.out
#SBATCH --error=jepa_adaptation/jobs/logs/downstream_1_%j.err

set -e

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
NOTEBOOK="$PROJECT_ROOT/jepa_adaptation/downstream/downstream_1.ipynb"
LOG_DIR="$PROJECT_ROOT/jepa_adaptation/jobs/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

module load cuda
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=== downstream_1 — $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=43200 \
    --ExecutePreprocessor.kernel_name=python3 \
    "$NOTEBOOK" 2>&1 | tee "$LOG_DIR/downstream_1_${SLURM_JOB_ID}_exec.log"

echo "=== terminé — $(date) ==="
