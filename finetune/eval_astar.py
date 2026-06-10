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

import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader

from finetune.dataset import ArcadeDataset
from finetune.encoder import MedVAEAutoencoder
from finetune.metrics.seg_metrics import SegMetrics
from finetune.models import build_unet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_a",     required=True,
                        help="Config condition A (condition_a.yaml)")
    parser.add_argument("--checkpoint_a", required=True,
                        help="Checkpoint condition A (best_model_condition_a_unet.pth)")
    parser.add_argument("--medvae_model",  default="medvae_4_1_2d")
    parser.add_argument("--medvae_modality", default="xray")
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
        batch_size=config["training"]["batch_size"],
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


if __name__ == "__main__":
    main()
