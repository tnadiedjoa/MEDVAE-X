#!/bin/bash
#SBATCH --job-name=jepa_recon
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=jepa_adaptation/jobs/logs/recon_%j.out
#SBATCH --error=jepa_adaptation/jobs/logs/recon_%j.err

set -e
# Folder the job was submitted from (run sbatch from the repository root)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
# Reconstruction (official test set) of the pre-trained MedVAE and of JEPA stages 1 and 2.
STAGE1="${STAGE1_CKPT:-jepa_adaptation/outputs/stage1/best.pt}"
STAGE2="${STAGE2_CKPT:-jepa_adaptation/outputs/stage2/best.pt}"
python jepa_adaptation/eval/reconstruction.py --model medvae
python jepa_adaptation/eval/reconstruction.py --model stage1 --ckpt "$STAGE1"
python jepa_adaptation/eval/reconstruction.py --model stage2 --ckpt "$STAGE2"
