#!/bin/bash
#SBATCH --job-name=im06_cond_c
#SBATCH --output=finetune/slurm/logs/train_c_%j.out
#SBATCH --error=finetune/slurm/logs/train_c_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"


cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Condition C — démarrage ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
date

CKPT="finetune/checkpoints/medvae_finetuned/best_medvae_finetuned.pth"
if [ ! -f "$CKPT" ]; then
  echo "ERREUR : checkpoint MedVAE introuvable : $CKPT"
  echo "Lancer finetune_medvae.sbatch avant train_c.sbatch"
  exit 1
fi

python -m finetune.train --config finetune/configs/condition_c.yaml

echo "=== Condition C — terminé ===" && date
