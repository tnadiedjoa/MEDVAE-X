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

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

# Linear probes (segments et vaisseaux) du MedVAE pré-entraîné et des étapes 1 et 2 JEPA.
# Checkpoints par défaut : jepa_adaptation/outputs/stage{1,2}/best.pt
STAGE1="${STAGE1_CKPT:-jepa_adaptation/outputs/stage1/best.pt}"
STAGE2="${STAGE2_CKPT:-jepa_adaptation/outputs/stage2/best.pt}"

for task in segments vessels; do
  python jepa_adaptation/eval/probe.py --model medvae --task $task
  python jepa_adaptation/eval/probe.py --model stage1 --ckpt "$STAGE1" --task $task
  python jepa_adaptation/eval/probe.py --model stage2 --ckpt "$STAGE2" --task $task
done
