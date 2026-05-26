#!/bin/bash
#SBATCH --job-name=vae_phase2_clip
#SBATCH --partition=H100,A100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=jepa-adaptation/jobs/logs/phase2_clip_%j.out
#SBATCH --error=jepa-adaptation/jobs/logs/phase2_clip_%j.err

set -e

PROJECT_ROOT=/home/infres/yrothlin-24/projet_IM06
MICROMAMBA=/home/infres/yrothlin-24/.local/bin/micromamba

mkdir -p "$PROJECT_ROOT/jepa-adaptation/jobs/logs"
cd "$PROJECT_ROOT"

module load cuda

eval "$($MICROMAMBA shell hook --shell bash)"
micromamba activate IM05

CKPT="$PROJECT_ROOT/jepa-adaptation/outputs/pretraining/phase1/best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Phase-1 checkpoint not found at $CKPT" >&2
    echo "Run pretraining_phase_1.sbatch first." >&2
    exit 1
fi

echo "=== Phase 2 (CLIP) — BiomedCLIP latent alignment ==="
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Ckpt   : $CKPT"
echo "Date   : $(date)"
echo "========================================"

python jepa-adaptation/pretraining.py \
    --config jepa-adaptation/configs/pretraining.yaml \
    --phase 2 \
    --mode clip

echo "=== Done : $(date) ==="
