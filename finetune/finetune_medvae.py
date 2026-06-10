import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from medvae import MVAE


class ArcadeImageDataset(Dataset):
    """Images ARCADE seules (sans masques) — pour fine-tuner MedVAE."""

    def __init__(self, images_dir: str, img_size: int = 512):
        self.images_dir = images_dir
        self.img_size   = img_size
        self.files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not self.files:
            raise FileNotFoundError(f"Aucune image trouvée dans : {images_dir}")
        print(f"Dataset : {len(self.files)} images dans {images_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = os.path.join(self.images_dir, self.files[idx])
        img  = np.array(
            Image.open(path).convert("L").resize((self.img_size, self.img_size))
        )
        # [0, 255] → [-1, 1]  (convention MedVAE)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0


def fine_tune(config: dict) -> None:
    enc_cfg   = config["encoder"]
    data_cfg  = config["data"]
    train_cfg = config["training"]
    log_cfg   = config["logging"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Charge MedVAE depuis HuggingFace ─────────────────────────────────
    print(f"Chargement de MedVAE ({enc_cfg['model_name']})...")
    mvae = MVAE(model_name=enc_cfg["model_name"], modality=enc_cfg["modality"]).to(device)

    # Dégèle tout pour le fine-tuning
    for p in mvae.parameters():
        p.requires_grad = True
    mvae.train()

    n_params = sum(p.numel() for p in mvae.parameters())
    print(f"MedVAE prêt — {n_params:,} paramètres entraînables")

    # ── Dataset ──────────────────────────────────────────────────────────
    full_ds = ArcadeImageDataset(
        images_dir=data_cfg["train_images"],
        img_size=data_cfg.get("img_size", 512),
    )
    n_val   = max(1, int(len(full_ds) * data_cfg.get("val_ratio", 0.1)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train : {n_train}  |  Val : {n_val}")

    loader_kw = dict(
        batch_size  = train_cfg["batch_size"],
        num_workers = data_cfg.get("num_workers", 4),
        pin_memory  = data_cfg.get("pin_memory", True),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = AdamW(
        mvae.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=train_cfg.get("lr_min", 1e-7),
    )

    # ── Sauvegarde ───────────────────────────────────────────────────────
    save_dir = log_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_medvae_finetuned.pth")

    best_val  = float("inf")
    patience  = train_cfg.get("early_stopping_patience", 10)
    no_imp    = 0
    history   = {"train": [], "val": []}

    # ── Boucle d'entraînement ────────────────────────────────────────────
    for epoch in range(1, train_cfg["epochs"] + 1):

        # — train —
        mvae.train()
        train_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            latent        = mvae.encode(images)
            reconstructed = mvae.decode(latent)

            loss = F.l1_loss(reconstructed, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                mvae.parameters(), train_cfg.get("grad_clip", 1.0)
            )
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # — val —
        mvae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                latent        = mvae.encode(images)
                reconstructed = mvae.decode(latent)
                val_loss += F.l1_loss(reconstructed, images).item()
        val_loss /= len(val_loader)

        scheduler.step()

        history["train"].append({"epoch": epoch, "loss": train_loss})
        history["val"].append({"epoch": epoch, "loss": val_loss})

        print(f"Epoch {epoch:3d}/{train_cfg['epochs']}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            no_imp   = 0
            torch.save({
                "state_dict": mvae.state_dict(),
                "epoch":      epoch,
                "val_loss":   val_loss,
                "config":     config,
            }, ckpt_path)
            print(f"  ✓ val_loss={best_val:.5f} — checkpoint sauvegardé")
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"\nEarly stopping à l'epoch {epoch}.")
                break

    # — historique —
    hist_path = os.path.join(save_dir, "medvae_finetune_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFine-tuning terminé.")
    print(f"Checkpoint : {ckpt_path}")
    print(f"→ Mettre à jour condition_c.yaml :")
    print(f"    checkpoint_path: \"{ckpt_path}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    fine_tune(config)


if __name__ == "__main__":
    main()
