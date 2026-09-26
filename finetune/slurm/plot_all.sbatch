#!/bin/bash
#SBATCH --job-name=im06_plots
#SBATCH --output=finetune/slurm/logs/plot_all_%j.out
#SBATCH --error=finetune/slurm/logs/plot_all_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -e

# Dossier depuis lequel le job a été soumis (lancer sbatch depuis la racine du repo)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"


cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Plots — démarrage ===" && date

RESULTS="finetune/results"
FIGS="finetune/figures"

# Courbes comparatives des résultats officiels du projet
python -m finetune.plot \
  --history \
    "$RESULTS/condition_a_unet_history.json" \
    "$RESULTS/condition_b_medvae_history.json" \
    "$RESULTS/condition_c_medvae_finetuned_history.json" \
    "$RESULTS/condition_d_unet_medvae_reconstructed_history.json" \
  --labels "Condition A" "Condition B" "Condition C" "Condition D" \
  --save_dir "$FIGS"

# Figures de prédictions, dans le dossier de chaque run passé en argument :
#   sbatch finetune/slurm/plot_all.sbatch experiments/runs/<run A> experiments/runs/<run B> ...
for RUN in "$@"; do
  NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['experiment']['name'])" "$RUN/config.yaml")
  python -m finetune.plot \
    --history "$RUN/${NAME}_history.json" \
    --labels "$NAME" --save_dir "$RUN" \
    --predict \
    --checkpoint "$RUN/best_model_${NAME}.pth" \
    --config "$RUN/config.yaml"
done

echo "=== Plots — terminé ===" && date
