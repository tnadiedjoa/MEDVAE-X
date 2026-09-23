#!/bin/bash
#SBATCH --job-name=jepa_stage2
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=jepa_adaptation/jobs/logs/stage2_%j.out
#SBATCH --error=jepa_adaptation/jobs/logs/stage2_%j.err

set -e

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p "$PROJECT_ROOT/jepa_adaptation/outputs/stage2/logs"
cd "$PROJECT_ROOT"

module load cuda
source "$PROJECT_ROOT/.venv/bin/activate"

CKPT="$PROJECT_ROOT/jepa_adaptation/outputs/stage1/best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: stage-1 checkpoint not found at $CKPT" >&2
    exit 1
fi

echo "=== stage 2 — source: ARCADE — $(date) ==="

python jepa_adaptation/stage2_training.py \
    --config jepa_adaptation/configs/stage_2.yaml \
    --model-config jepa_adaptation/configs/model.yaml
