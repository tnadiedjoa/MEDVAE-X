#!/bin/bash
#SBATCH --job-name=im06_plots
#SBATCH --output=finetune/slurm/logs/plot_all_%j.out
#SBATCH --error=finetune/slurm/logs/plot_all_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"


cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Plots — démarrage ===" && date

CKPTS="finetune/checkpoints"
FIGS="finetune/figures"

python -m finetune.plot \
  --history "$CKPTS/condition_a_unet_history.json" \
  --labels "Condition A" --save_dir "$FIGS" \
  --predict \
  --checkpoint "$CKPTS/best_model_condition_a_unet.pth" \
  --config finetune/configs/condition_a.yaml

python -m finetune.plot \
  --history "$CKPTS/condition_b_medvae_history.json" \
  --labels "Condition B" --save_dir "$FIGS" \
  --predict \
  --checkpoint "$CKPTS/best_model_condition_b_medvae.pth" \
  --config finetune/configs/condition_b.yaml

python -m finetune.plot \
  --history "$CKPTS/condition_c_medvae_finetuned_history.json" \
  --labels "Condition C" --save_dir "$FIGS" \
  --predict \
  --checkpoint "$CKPTS/best_model_condition_c_medvae_finetuned.pth" \
  --config finetune/configs/condition_c.yaml

python -m finetune.plot \
  --history "$CKPTS/condition_d_unet_medvae_reconstructed_history.json" \
  --labels "Condition D" --save_dir "$FIGS" \
  --predict \
  --checkpoint "$CKPTS/best_model_condition_d_unet_medvae_reconstructed.pth" \
  --config finetune/configs/condition_d.yaml

python -m finetune.plot \
  --history \
    "$CKPTS/condition_a_unet_history.json" \
    "$CKPTS/condition_b_medvae_history.json" \
    "$CKPTS/condition_c_medvae_finetuned_history.json" \
    "$CKPTS/condition_d_unet_medvae_reconstructed_history.json" \
  --labels "Condition A" "Condition B" "Condition C" "Condition D" \
  --save_dir "$FIGS"

echo "=== Plots — terminé ===" && date
