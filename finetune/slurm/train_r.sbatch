#!/bin/bash
#SBATCH --job-name=im06_cond_r
#SBATCH --output=finetune/slurm/logs/train_b_%j.out
#SBATCH --error=finetune/slurm/logs/train_b_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

# Folder the job was submitted from (run sbatch from the repository root)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"


cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Condition R (E16 control) — start ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
date

python -m finetune.train --config finetune/configs/condition_r.yaml "$@"

echo "=== Condition R (E16 control) — done ===" && date
