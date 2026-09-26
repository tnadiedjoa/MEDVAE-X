#!/bin/bash
#SBATCH --job-name=im06_cond_a
#SBATCH --output=finetune/slurm/logs/train_a_%j.out
#SBATCH --error=finetune/slurm/logs/train_a_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"


cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Condition A — démarrage ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
date

python -m finetune.train --config finetune/configs/condition_a.yaml "$@"

echo "=== Condition A — terminé ===" && date
