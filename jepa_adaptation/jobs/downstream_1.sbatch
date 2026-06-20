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

PROJECT_ROOT=/home/infres/yrothlin-24/projet_IM06
MICROMAMBA=/home/infres/yrothlin-24/.local/bin/micromamba
NOTEBOOK="$PROJECT_ROOT/jepa_adaptation/downstream/downstream_1.ipynb"
LOG_DIR="$PROJECT_ROOT/jepa_adaptation/jobs/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

module load cuda
eval "$($MICROMAMBA shell hook --shell bash)"
micromamba activate IM05

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
