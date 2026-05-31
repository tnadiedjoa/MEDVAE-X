import argparse
import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import yaml


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

    # Loss totale (Dice + CE)
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

    # Dice loss
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

    # Dice Mean (métrique principale)
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

    # learning rate
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

    # comparaison finale des 3 pipelines sur la métrique principale (Dice val)
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
    x     = range(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.suptitle("Dice par classe (artères coronaires)", fontsize=14, fontweight="bold")

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

    ax.set_xlabel("Classe (artère)"); ax.set_ylabel("Dice")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"C{i}" for i in range(num_classes)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_per_class.png"), dpi=150)
    plt.close()
    print("Sauvegardé : dice_per_class.png")


def plot_predictions(
    checkpoint_path: str,
    config_path: str,
    save_dir: str,
    n_samples: int = 4,
    seed: int = 42,
) -> None:
    from finetune.dataset import ArcadeDataset, split_dataset
    from finetune.models import build_unet

    # Charge la config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charge le modèle
    model = build_unet(config["model"])
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Checkpoint chargé : epoch {ckpt['epoch']}, Dice {ckpt['dice']:.4f}")

    # Charge le test set
    root = data_cfg.get("data_root", "")
    def full(p):
        return os.path.join(root, p) if root else p

    test_dataset = ArcadeDataset(
        images_dir=full(data_cfg["val_images"]),
        annotations=full(data_cfg["val_ann"]),
    )

    # Tirage aléatoire reproductible
    random.seed(seed)
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))

    fig, axes = plt.subplots(
        n_samples, 3,
        figsize=(12, 4 * n_samples)
    )
    # Assure que axes est toujours 2D
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Titres des colonnes
    for ax, title in zip(axes[0], ["Image originale", "Masque GT", "Masque prédit"]):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for row, idx in enumerate(indices):
        image, mask_gt = test_dataset[idx]
        image_input    = image.unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            logits    = model(image_input)
            mask_pred = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Image originale (niveaux de gris)
        axes[row, 0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 0].axis("off")

        # Masque GT
        axes[row, 1].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 1].imshow(mask_gt.numpy(), cmap="tab20",
                            alpha=0.6, vmin=0, vmax=25)
        axes[row, 1].axis("off")

        # Masque prédit
        axes[row, 2].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 2].imshow(mask_pred, cmap="tab20",
                            alpha=0.6, vmin=0, vmax=25)
        axes[row, 2].axis("off")

    condition = config["experiment"]["name"]
    fig.suptitle(f"Visualisation des segmentations — {condition}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"predictions_{condition}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : predictions_{condition}.png")


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
    # Arguments pour la visualisation des segmentations
    parser.add_argument(
        "--predict", action="store_true",
        help="Active la visualisation des segmentations"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Chemin vers best_model.pth (requis si --predict)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Chemin vers le yaml de la condition (requis si --predict)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="Nombre d'images à visualiser (défaut : 4)"
    )
    args = parser.parse_args()

    # Labels par défaut si non fournis
    if args.labels is None:
        args.labels = [os.path.basename(p).replace("_history.json", "")
                       for p in args.history]

    if len(args.history) != len(args.labels):
        raise ValueError("Le nombre de --labels doit correspondre au nombre de --history")

    # Courbes d'entraînement
    histories = [load_history(p) for p in args.history]
    plot_all(histories, args.labels, args.save_dir)

    # Dice par classe 
    results_path = os.path.join(
        os.path.dirname(args.history[0]), "results.json"
    )
    if os.path.exists(results_path):
        plot_dice_per_class(results_path, args.save_dir)

    # Visualisation des segmentations
    if args.predict:
        if args.checkpoint is None or args.config is None:
            raise ValueError("--checkpoint et --config sont requis avec --predict")
        plot_predictions(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            save_dir=args.save_dir,
            n_samples=args.n_samples,
        )

    print(f"\nToutes les figures sont dans : {args.save_dir}/")


if __name__ == "__main__":
    main()