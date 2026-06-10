#!/bin/bash
#SBATCH --job-name=im06_cond_d
#SBATCH --output=finetune/slurm/logs/train_d_%j.out
#SBATCH --error=finetune/slurm/logs/train_d_%j.err
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ── À adapter ────────────────────────────────────────────────────────────
PROJECT_DIR="/home/infres/tpalagi-24/projet_IM06"

# ─────────────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Condition D — démarrage ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
date

python -m finetune.train --config finetune/configs/condition_d.yaml

echo "=== Condition D — terminé ===" && date
