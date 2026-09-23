#!/bin/bash
#SBATCH -p 3090
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=jepa_adaptation/jobs/logs/stage1_%j.out
#SBATCH --error=jepa_adaptation/jobs/logs/stage1_%j.err

set -e

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p "$PROJECT_ROOT/jepa_adaptation/outputs/stage1/logs"
cd "$PROJECT_ROOT"

module load cuda
source "$PROJECT_ROOT/.venv/bin/activate"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=== stage 1 — source: ARCADE — $(date) ==="

python jepa_adaptation/stage1_training.py \
    --config jepa_adaptation/configs/stage_1.yaml \
    --model-config jepa_adaptation/configs/model.yaml
