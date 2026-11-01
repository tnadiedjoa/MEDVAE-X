#!/bin/bash
#SBATCH --job-name=jepa_probe
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=jepa_adaptation/jobs/logs/probe_%j.out
#SBATCH --error=jepa_adaptation/jobs/logs/probe_%j.err

set -e

# Folder the job was submitted from (run sbatch from the repository root)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

# Vessel / background linear probe of the pre-trained MedVAE and of JEPA stages 1 and 2, published
# setting (E19): 100 epochs, lr 1e-3 and 1e-2, the lr being chosen afterwards on val_dice_fg.
#   STAGE1_CKPT=experiments/runs/<stage-1 run>/best.pt STAGE2_CKPT=experiments/runs/<stage-2 run>/best.pt \
#   OUT=experiments/jepa/<name> sbatch jepa_adaptation/jobs/probe.sbatch
STAGE1="${STAGE1_CKPT:?STAGE1_CKPT: stage-1 checkpoint}"
STAGE2="${STAGE2_CKPT:?STAGE2_CKPT: stage-2 checkpoint}"
OUT="${OUT:-experiments/jepa/probe_$(date +%Y-%m-%d_%H%M%S)}"
mkdir -p "$OUT"

for lr in 1e-3 1e-2; do
  python jepa_adaptation/eval/probe.py --model medvae --task vessels --epochs 100 --lr $lr --out "$OUT/probe_medvae_lr$lr.json"
  python jepa_adaptation/eval/probe.py --model stage1 --ckpt "$STAGE1" --task vessels --epochs 100 --lr $lr --out "$OUT/probe_stage1_lr$lr.json"
  python jepa_adaptation/eval/probe.py --model stage2 --ckpt "$STAGE2" --task vessels --epochs 100 --lr $lr --out "$OUT/probe_stage2_lr$lr.json"
done
