#!/bin/bash
#SBATCH --job-name=im06_astar
#SBATCH --output=finetune/slurm/logs/eval_astar_%j.out
#SBATCH --error=finetune/slurm/logs/eval_astar_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=H100,A100,L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"


cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Condition A* — démarrage ==="
echo "Job ID : $SLURM_JOB_ID"
date

python -m finetune.eval_astar \
  --config_a     finetune/configs/condition_a.yaml \
  --checkpoint_a finetune/checkpoints/best_model_condition_a_unet.pth

echo "=== Condition A* — terminé ===" && date
