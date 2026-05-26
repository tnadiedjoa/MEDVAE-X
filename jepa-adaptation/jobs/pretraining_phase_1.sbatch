#!/bin/bash
#SBATCH --job-name=vae_phase1
#SBATCH --partition=H100,A100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=jepa-adaptation/jobs/logs/phase1_%j.out
#SBATCH --error=jepa-adaptation/jobs/logs/phase1_%j.err

set -e

PROJECT_ROOT=/home/infres/yrothlin-24/projet_IM06
MICROMAMBA=/home/infres/yrothlin-24/.local/bin/micromamba

mkdir -p "$PROJECT_ROOT/jepa-adaptation/jobs/logs"
cd "$PROJECT_ROOT"

module load cuda

eval "$($MICROMAMBA shell hook --shell bash)"
micromamba activate IM05

echo "=== Phase 1 — VAE pretraining ==="
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date   : $(date)"
echo "========================================"

python jepa-adaptation/pretraining.py \
    --config jepa-adaptation/configs/pretraining.yaml \
    --phase 1

echo "=== Done : $(date) ==="
