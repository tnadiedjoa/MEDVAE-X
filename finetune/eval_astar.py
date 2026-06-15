"""
Condition A* : U-Net de la condition A évalué sur des images
reconstruites par MedVAE (encode→decode), sans réentraînement.

Mesure la dégradation brute de la compression sur un U-Net non adapté.
Comparer avec condition D (U-Net adapté) pour quantifier le gain de l'adaptation.

Lancer depuis la racine projet_IM06/ :
  python -m finetune.eval_astar \
    --config_a finetune/configs/condition_a.yaml \
    --checkpoint_a finetune/checkpoints/best_model_condition_a_unet.pth
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader

from finetune.dataset import ArcadeDataset
from finetune.encoder import MedVAEAutoencoder
from finetune.metrics.seg_metrics import SegMetrics
from finetune.models import build_unet


def plot_predictions_astar(unet, autoencoder, test_dataset, device, save_dir,
                           n_samples=4, seed=42):
    random.seed(seed)
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for ax, title in zip(axes[0], ["Image originale", "Masque GT", "Masque prédit (A*)"]):
        ax.set_title(title, fontsize=12, fontweight="bold")

    use_amp = device.type == "cuda"
    for row, idx in enumerate(indices):
        image, mask_gt = test_dataset[idx]
        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed = autoencoder(image_input)
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = unet(reconstructed)
            mask_pred = logits.argmax(dim=1).squeeze().cpu().numpy()

        axes[row, 0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 1].imshow(mask_gt.numpy(), cmap="tab20", alpha=0.6, vmin=0, vmax=25)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(reconstructed.squeeze().cpu().numpy(), cmap="gray")
        axes[row, 2].imshow(mask_pred, cmap="tab20", alpha=0.6, vmin=0, vmax=25)
        axes[row, 2].axis("off")

    fig.suptitle("Condition A* — U-Net A sur images MedVAE reconstituées",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "predictions_condition_astar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_a",     required=True,
                        help="Config condition A (condition_a.yaml)")
    parser.add_argument("--checkpoint_a", required=True,
                        help="Checkpoint condition A (best_model_condition_a_unet.pth)")
    parser.add_argument("--medvae_model",  default="medvae_4_1_2d")
    parser.add_argument("--medvae_modality", default="xray")
    parser.add_argument("--predict", action="store_true",
                        help="Génère la visualisation des prédictions A*")
    parser.add_argument("--save_dir", default="finetune/figures",
                        help="Dossier de sauvegarde des figures (défaut : finetune/figures)")
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    with open(args.config_a) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── U-Net condition A (gelé — on l'évalue, pas on l'entraîne) ────────
    unet = build_unet(config["model"])
    ckpt = torch.load(args.checkpoint_a, map_location=device)
    unet.load_state_dict(ckpt["model"])
    unet.to(device).eval()
    for p in unet.parameters():
        p.requires_grad = False
    print(f"Checkpoint A chargé : epoch {ckpt['epoch']}, Dice A = {ckpt['dice']:.4f}")

    # ── MedVAE gelé comme préprocesseur ──────────────────────────────────
    autoencoder = MedVAEAutoencoder(
        model_name=args.medvae_model,
        modality=args.medvae_modality,
        device=device,
    )

    # ── Test dataset ─────────────────────────────────────────────────────
    data_cfg = config["data"]
    test_dataset = ArcadeDataset(
        images_dir=data_cfg["val_images"],
        annotations=data_cfg["val_ann"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,  # MedVAE attention OOM à batch_size=8
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )
    print(f"Test set : {len(test_dataset)} images")

    # ── Évaluation ───────────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    metrics = SegMetrics(num_classes=data_cfg["num_classes"], device=device)

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)

            reconstructed = autoencoder(images)   # encode→decode MedVAE

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = unet(reconstructed)

            metrics.update(logits, masks)

    results = metrics.compute()
    print(f"\n=== Résultats A* (U-Net A sur images MedVAE reconstituées) ===")
    print(f"Dice mean : {results['dice_mean']:.4f}")
    print(f"IoU  mean : {results['iou_mean']:.4f}")
    print(f"\nRappel : Dice A (images originales) = {ckpt['dice']:.4f}")
    print(f"  → dégradation brute MedVAE : {ckpt['dice'] - results['dice_mean']:.4f}")

    # ── Sauvegarde dans results.json (même fichier que les autres conditions)
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["condition_astar"] = {
        "dice_mean":      results["dice_mean"],
        "iou_mean":       results["iou_mean"],
        "dice_per_class": results["dice_per_class"],
    }

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRésultats sauvegardés dans : {results_path}")

    if args.predict:
        plot_predictions_astar(
            unet=unet,
            autoencoder=autoencoder,
            test_dataset=test_dataset,
            device=device,
            save_dir=args.save_dir,
            n_samples=args.n_samples,
        )


if __name__ == "__main__":
    main()
