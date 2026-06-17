#!/bin/bash
#SBATCH -p H100,A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=jepa_adaptation/outputs/stage1/logs/stage1_%j.out
#SBATCH --error=jepa_adaptation/outputs/stage1/logs/stage1_%j.err

set -e

PROJECT_ROOT=/home/infres/yrothlin-24/projet_IM06
MICROMAMBA=/home/infres/yrothlin-24/.local/bin/micromamba

mkdir -p "$PROJECT_ROOT/jepa_adaptation/outputs/stage1/logs"
cd "$PROJECT_ROOT"

module load cuda
eval "$($MICROMAMBA shell hook --shell bash)"
micromamba activate IM05

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=== stage 1 — source: ARCADE — $(date) ==="

python jepa_adaptation/stage1_training.py \
    --config jepa_adaptation/configs/stage_1.yaml \
    --model-config jepa_adaptation/configs/model.yaml
