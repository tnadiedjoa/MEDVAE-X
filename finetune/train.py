import argparse
import random
import os

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from finetune.dataset import ArcadeDataset, split_dataset
from finetune.models import build_unet, build_seg_head
from finetune.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CPU uniquement")
    return device


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]

    train_ids, val_ids = split_dataset(
        data_cfg["train_ann"],
        train_ratio=data_cfg["train_ratio"],
        seed=config["experiment"]["seed"],
    )

    train_dataset = ArcadeDataset(data_cfg["train_images"], data_cfg["train_ann"],
                                  train_ids, augment=True)
    val_dataset   = ArcadeDataset(data_cfg["train_images"], data_cfg["train_ann"],
                                  val_ids,   augment=False)
    test_dataset  = ArcadeDataset(data_cfg["val_images"],   data_cfg["val_ann"],
                                  augment=False)

    print(f"Train : {len(train_dataset)} images")
    print(f"Val   : {len(val_dataset)} images")
    print(f"Test  : {len(test_dataset)} images")

    loader_kwargs = dict(
        batch_size  = config["training"]["batch_size"],
        num_workers = data_cfg["num_workers"],
        pin_memory  = data_cfg["pin_memory"],
    )

    return (
        DataLoader(train_dataset, shuffle=True,  **loader_kwargs),
        DataLoader(val_dataset,   shuffle=False, **loader_kwargs),
        DataLoader(test_dataset,  shuffle=False, **loader_kwargs),
    )


def build_model(config: dict, device: torch.device):
    """
    Instancie le bon modèle selon la condition :
    - Condition A : U-Net complet
    - Condition B/C : encodeur MedVAE gelé + tête de segmentation
    """
    condition = config["experiment"]["condition"]

    if condition == "A":
        model = build_unet(config["model"])

    elif condition in ("B", "C"):
        from finetune.encoder import MedVAEEncoder
        from finetune.models import build_seg_head
        import torch.nn as nn

        encoder = MedVAEEncoder(
            model_name=config["encoder"]["model_name"],
            modality=config["encoder"]["modality"],
            device=device,
        )
        seg_head = build_seg_head(config["model"])

        # Combine encodeur + tête dans un seul module
        class EncoderWithHead(nn.Module):
            def __init__(self, enc, head):
                super().__init__()
                self.encoder  = enc
                self.seg_head = head

            def forward(self, x):
                latent = self.encoder.encode(x)
                return self.seg_head(latent)

        model = EncoderWithHead(encoder, seg_head)

    else:
        raise ValueError(f"Condition inconnue : {condition}")

    # Compte uniquement les paramètres entraînables
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n_params:,}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Expérience : {config['experiment']['name']}")

    set_seed(config["experiment"]["seed"])
    device = get_device()

    train_loader, val_loader, test_loader = build_dataloaders(config)
    model   = build_model(config, device)
    trainer = Trainer(model=model, config=config, device=device)
    trainer.fit(train_loader, val_loader)
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
