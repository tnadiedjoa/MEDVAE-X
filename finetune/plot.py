import argparse
import json
import os
import matplotlib.pyplot as plt


def load_history(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_all(histories: list[dict], labels: list[str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Couleurs distinctes par condition
    colors = ["#2E86C1", "#E74C3C", "#27AE60", "#8E44AD"]

    def get_values(history, split, key):
        return [epoch[key] for epoch in history[split]]

    def get_epochs(history, split):
        return [epoch["epoch"] for epoch in history[split]]

    # Figure 1 : Loss totale 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Loss totale (Dice + CE)", fontsize=14, fontweight="bold")

    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        epochs_train = get_epochs(history, "train")
        epochs_val   = get_epochs(history, "val")

        axes[0].plot(epochs_train, get_values(history, "train", "loss"),
                     color=color, label=label, linewidth=2)
        axes[1].plot(epochs_val, get_values(history, "val", "loss"),
                     color=color, label=label, linewidth=2)

    axes[0].set_title("Train"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[0].legend(); axes[1].legend()
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_totale.png"), dpi=150)
    plt.close()
    print("Sauvegardé : loss_totale.png")

    #  Figure 2 : Dice Loss 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dice Loss", fontsize=14, fontweight="bold")

    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        axes[0].plot(get_epochs(history, "train"),
                     get_values(history, "train", "dice_loss"),
                     color=color, label=label, linewidth=2)
        axes[1].plot(get_epochs(history, "val"),
                     get_values(history, "val", "dice_loss"),
                     color=color, label=label, linewidth=2)

    axes[0].set_title("Train"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Dice Loss")
    axes[1].set_title("Validation"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice Loss")
    axes[0].legend(); axes[1].legend()
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_loss.png"), dpi=150)
    plt.close()
    print("Sauvegardé : dice_loss.png")

    # Figure 3 : Dice mean (métrique principale) 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dice Mean — métrique principale", fontsize=14, fontweight="bold")

    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        axes[0].plot(get_epochs(history, "train"),
                     get_values(history, "train", "dice_mean"),
                     color=color, label=label, linewidth=2)
        axes[1].plot(get_epochs(history, "val"),
                     get_values(history, "val", "dice_mean"),
                     color=color, label=label, linewidth=2)

    axes[0].set_title("Train"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Dice")
    axes[1].set_title("Validation"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1)
    axes[0].legend(); axes[1].legend()
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_mean.png"), dpi=150)
    plt.close()
    print("Sauvegardé : dice_mean.png")

    # ── Figure 4 : Learning rate ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Learning Rate", fontsize=14, fontweight="bold")

    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        ax.plot(get_epochs(history, "train"),
                get_values(history, "train", "lr"),
                color=color, label=label, linewidth=2)

    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_rate.png"), dpi=150)
    plt.close()
    print("Sauvegardé : learning_rate.png")

    # Figure 5 : Comparaison finale A / B / C (val Dice) 
    if len(histories) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Comparaison Dice val — Conditions A / B / C",
                     fontsize=14, fontweight="bold")

        for i, (history, label) in enumerate(zip(histories, labels)):
            color = colors[i % len(colors)]
            ax.plot(get_epochs(history, "val"),
                    get_values(history, "val", "dice_mean"),
                    color=color, label=label, linewidth=2)

        ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Mean (val)")
        ax.set_ylim(0, 1)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "comparaison_conditions.png"), dpi=150)
        plt.close()
        print("Sauvegardé : comparaison_conditions.png")

def plot_dice_per_class(results_path: str, save_dir: str) -> None:
    with open(results_path, "r") as f:
        results = json.load(f)

    colors = {
        "condition_a": "#2E86C1",
        "condition_b": "#E74C3C",
        "condition_c": "#27AE60",
    }
    labels = {
        "condition_a": "Condition A",
        "condition_b": "Condition B",
        "condition_c": "Condition C",
    }

    num_classes = len(list(results.values())[0]["dice_per_class"])
    x = range(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.suptitle("Dice par classe", fontsize=14, fontweight="bold")

    for i, (key, data) in enumerate(results.items()):
        offset = (i - len(results) / 2) * width
        ax.bar(
            [xi + offset for xi in x],
            data["dice_per_class"],
            width=width,
            color=colors.get(key, "gray"),
            label=labels.get(key, key),
            alpha=0.85,
        )

    ax.set_xlabel("Classe (artère)")
    ax.set_ylabel("Dice")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"C{i}" for i in range(num_classes)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_per_class.png"), dpi=150)
    plt.close()
    print("Sauvegardé : dice_per_class.png")


def plot_predictions(
    model,
    dataset,
    device: torch.device,
    save_dir: str,
    n_samples: int = 4,
) -> None:
    import random
    import numpy as np

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    n_cols  = 2 + len(model)   # image + GT + une colonne par condition

    fig, axes = plt.subplots(
        n_samples, n_cols,
        figsize=(4 * n_cols, 4 * n_samples)
    )
    # Assure que axes est toujours 2D
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Masque GT"] + [f"Prédit {k}" for k in model.keys()]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_input = image.unsqueeze(0).to(device)  # (1, 1, H, W)

        # Image originale
        axes[row, 0].imshow(image.squeeze().cpu(), cmap="gray")
        axes[row, 0].axis("off")

        # Masque GT
        axes[row, 1].imshow(mask.cpu(), cmap="tab20", vmin=0, vmax=25)
        axes[row, 1].axis("off")

        # Prédictions par condition
        for col, (cond, m) in enumerate(model.items(), start=2):
            m.eval()
            with torch.no_grad():
                logits = m(image_input)
                pred   = logits.argmax(dim=1).squeeze().cpu()
            axes[row, col].imshow(pred, cmap="tab20", vmin=0, vmax=25)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictions.png"), dpi=150)
    plt.close()
    print("Sauvegardé : predictions.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history", nargs="+", required=True,
        help="Chemins vers les fichiers history.json"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels pour la légende (un par fichier history)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./figures",
        help="Dossier de sauvegarde des figures"
    )
    args = parser.parse_args()

    # Labels par défaut si non fournis
    if args.labels is None:
        args.labels = [os.path.basename(p).replace("_history.json", "")
                       for p in args.history]

    if len(args.history) != len(args.labels):
        raise ValueError("Le nombre de --labels doit correspondre au nombre de --history")

    histories = [load_history(p) for p in args.history]
    plot_all(histories, args.labels, args.save_dir)
    # Bar chart Dice par classe (lit results.json)
    results_path = os.path.join(os.path.dirname(args.history[0]), "results.json")
    if os.path.exists(results_path):
        plot_dice_per_class(results_path, args.save_dir)
    else:
        print("results.json introuvable — plot_dice_per_class ignoré")
    print(f"\nToutes les figures sont dans : {args.save_dir}/")


if __name__ == "__main__":
    main()
