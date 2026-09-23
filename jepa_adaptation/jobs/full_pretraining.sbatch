#!/bin/bash
#SBATCH --job-name=jepa_full
#SBATCH --partition=H100,A100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=jepa_adaptation/jobs/logs/full_%j.out
#SBATCH --error=jepa_adaptation/jobs/logs/full_%j.err

# ── paramètres ────────────────────────────────────────────────
EPOCHS_STAGE1=30
EPOCHS_STAGE2=30
# ──────────────────────────────────────────────────────────────

set -e

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p "$PROJECT_ROOT/jepa_adaptation/outputs/stage1/logs"
mkdir -p "$PROJECT_ROOT/jepa_adaptation/outputs/stage2/logs"
mkdir -p "$PROJECT_ROOT/jepa_adaptation/outputs/full_pretraining/logs"
cd "$PROJECT_ROOT"

module load cuda
source "$PROJECT_ROOT/.venv/bin/activate"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# ── stage 1 : reconstruction ──────────────────────────────────
echo ""
echo "=== STAGE 1 (${EPOCHS_STAGE1} epochs) — $(date) ==="

python jepa_adaptation/stage1_training.py \
    --config       jepa_adaptation/configs/stage_1.yaml \
    --model-config jepa_adaptation/configs/model.yaml \
    --epochs       "$EPOCHS_STAGE1"

CKPT="$PROJECT_ROOT/jepa_adaptation/outputs/stage1/best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: stage-1 checkpoint introuvable ($CKPT)" >&2
    exit 1
fi
echo "=== STAGE 1 terminé — checkpoint : $CKPT ==="

# ── stage 2 : JEPA ───────────────────────────────────────────
echo ""
echo "=== STAGE 2 (${EPOCHS_STAGE2} epochs) — $(date) ==="

python jepa_adaptation/stage2_training.py \
    --config       jepa_adaptation/configs/stage_2.yaml \
    --model-config jepa_adaptation/configs/model.yaml \
    --epochs       "$EPOCHS_STAGE2"

echo "=== STAGE 2 terminé — $(date) ==="
echo "Checkpoints stage2 : $PROJECT_ROOT/jepa_adaptation/outputs/stage2/"
