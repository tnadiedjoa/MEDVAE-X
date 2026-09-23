#!/bin/bash
#SBATCH --job-name=im06_medvae_ft
#SBATCH --output=finetune/slurm/logs/finetune_medvae_%j.out
#SBATCH --error=finetune/slurm/logs/finetune_medvae_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"


cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Fine-tuning MedVAE — démarrage ==="
echo "Job ID : $SLURM_JOB_ID"
date

python finetune/finetune_medvae.py --config finetune/configs/medvae_finetune.yaml

echo "=== Fine-tuning MedVAE — terminé ===" && date
