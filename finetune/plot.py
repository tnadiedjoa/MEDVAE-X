import argparse
import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


def load_history(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_all(histories: list[dict], labels: list[str], save_dir: str) -> None:

    os.makedirs(save_dir, exist_ok=True)

    # One colour per condition
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
    print("Saved: loss_totale.png")

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
    print("Saved: dice_loss.png")

    # Dice mean (main metric)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dice mean — main metric", fontsize=14, fontweight="bold")

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
    print("Saved: dice_mean.png")

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
    print("Saved: learning_rate.png")

    # final comparison of the pipelines on the main metric (validation Dice)
    if len(histories) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Comparaison Dice val — Conditions A / B / C / D",
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
        print("Saved: comparaison_conditions.png")


def plot_dice_per_class(results_path: str, save_dir: str) -> None:
    with open(results_path, "r") as f:
        results = json.load(f)

    colors = {
        "condition_a":     "#2E86C1",
        "condition_astar": "#85C1E9",
        "condition_b":     "#E74C3C",
        "condition_c":     "#27AE60",
        "condition_d":     "#8E44AD",
    }
    labels = {
        "condition_a":     "Condition A — U-Net original",
        "condition_astar": "Condition A* — U-Net A on MedVAE reconstructions (no adaptation)",
        "condition_b":     "Condition B — frozen MedVAE + head",
        "condition_c":     "Condition C — fine-tuned MedVAE + head",
        "condition_d":     "Condition D — U-Net adapted to the MedVAE reconstruction",
    }

    num_classes = len(list(results.values())[0]["dice_per_class"])
    x     = range(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.suptitle("Dice per class (coronary arteries)", fontsize=14, fontweight="bold")

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

    ax.set_xlabel("Class (artery)"); ax.set_ylabel("Dice")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"C{i}" for i in range(num_classes)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_per_class.png"), dpi=150)
    plt.close()
    print("Saved: dice_per_class.png")


def plot_predictions(
    checkpoint_path: str,
    config_path: str,
    save_dir: str,
    n_samples: int = 4,
    seed: int = 42,
) -> None:
    from finetune.config import load_config
    from finetune.dataset import ArcadeDataset
    from finetune.train import build_model

    config = load_config(config_path)

    data_cfg = config["data"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the exact architecture of the condition, then load the weights
    model = build_model(config, device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Checkpoint loaded: epoch {ckpt['epoch']}, Dice {ckpt['dice']:.4f}")

    # Load the test set (paths resolved against ARCADE_ROOT)
    test_dataset = ArcadeDataset(
        images_dir=data_cfg["val_images"],
        annotations=data_cfg["val_ann"],
    )

    # Reproducible random draw
    random.seed(seed)
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))

    fig, axes = plt.subplots(
        n_samples, 3,
        figsize=(12, 4 * n_samples)
    )
    # Make sure axes is always 2D
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Column titles
    for ax, title in zip(axes[0], ["Original image", "Ground-truth mask", "Predicted mask"]):
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

        # Predicted mask
        axes[row, 2].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 2].imshow(mask_pred, cmap="tab20",
                            alpha=0.6, vmin=0, vmax=25)
        axes[row, 2].axis("off")

    condition = config["experiment"]["name"]
    fig.suptitle(f"Segmentations — {condition}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"predictions_{condition}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: predictions_{condition}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history", nargs="+", required=True,
        help="Paths to the history.json files"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Legend labels (one per history file)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="finetune/figures",
        help="Output folder of the figures"
    )
    # Arguments of the segmentation visualisation
    parser.add_argument(
        "--predict", action="store_true",
        help="Enable the segmentation visualisation"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Chemin vers best_model.pth (requis si --predict)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to the yaml of the condition (required with --predict)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="Number of images to show (default: 4)"
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Only generate the prediction figures (no curves)"
    )
    args = parser.parse_args()

    predict_only = getattr(args, "predict_only", False)

    if not predict_only:
        # Default labels if none are given
        if args.labels is None:
            args.labels = [os.path.basename(p).replace("_history.json", "")
                           for p in args.history]

        if len(args.history) != len(args.labels):
            raise ValueError("The number of --labels must match the number of --history")

        # Training curves
        histories = [load_history(p) for p in args.history]
        plot_all(histories, args.labels, args.save_dir)

        # Dice per class
        results_path = os.path.join(
            os.path.dirname(args.history[0]), "results.json"
        )
        if os.path.exists(results_path):
            plot_dice_per_class(results_path, args.save_dir)

    # Segmentation figures
    if args.predict or predict_only:
        if args.checkpoint is None or args.config is None:
            raise ValueError("--checkpoint and --config are required with --predict")
        plot_predictions(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            save_dir=args.save_dir,
            n_samples=args.n_samples,
        )

    print(f"\nAll figures are in: {args.save_dir}/")


if __name__ == "__main__":
    main()