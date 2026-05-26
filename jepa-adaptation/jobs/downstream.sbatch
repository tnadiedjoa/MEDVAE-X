#!/bin/bash
#SBATCH --job-name=downstream_seg
#SBATCH --partition=H100,A100,L40S,A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=jepa-adaptation/jobs/logs/downstream_%j.out
#SBATCH --error=jepa-adaptation/jobs/logs/downstream_%j.err

set -e

PROJECT_ROOT=/home/infres/yrothlin-24/projet_IM06
MICROMAMBA=/home/infres/yrothlin-24/.local/bin/micromamba

mkdir -p "$PROJECT_ROOT/jepa-adaptation/jobs/logs"
cd "$PROJECT_ROOT"

module load cuda

eval "$($MICROMAMBA shell hook --shell bash)"
micromamba activate IM05

JEPA_CKPT="$PROJECT_ROOT/jepa-adaptation/outputs/pretraining/phase2_jepa/best.pt"
if [ ! -f "$JEPA_CKPT" ]; then
    echo "WARNING: JEPA checkpoint not found at $JEPA_CKPT" >&2
    echo "Make sure pretraining_phase_2_jepa.sbatch completed successfully." >&2
fi

echo "=== Downstream segmentation evaluation ==="
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date   : $(date)"
echo "========================================"

python jepa-adaptation/downstream.py \
    --config jepa-adaptation/configs/downstream.yaml

echo "=== Done : $(date) ==="
