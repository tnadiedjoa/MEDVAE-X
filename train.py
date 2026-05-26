import argparse
import random
import os

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from finetune.dataset import ArcadeDataset, split_dataset
from finetune.models import build_unet
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




def build_dataloaders(config):
    data_cfg = config["data"]

    # Split train/val sur le jeu d'entraînement
    train_ids, val_ids = split_dataset(
        data_cfg["train_ann"],
        train_ratio=data_cfg["train_ratio"],
        seed=config["experiment"]["seed"],
    )

    train_dataset = ArcadeDataset(data_cfg["train_images"], data_cfg["train_ann"], train_ids)
    val_dataset   = ArcadeDataset(data_cfg["train_images"], data_cfg["train_ann"], val_ids)
    test_dataset  = ArcadeDataset(data_cfg["val_images"],   data_cfg["val_ann"])

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Expérience : {config['experiment']['name']}")

    set_seed(config["experiment"]["seed"])
    device = get_device()

    train_loader, val_loader, test_loader = build_dataloaders(config)

    model = build_unet(config["model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres entraînables : {n_params:,}")

    trainer = Trainer(model=model, config=config, device=device)
    trainer.fit(train_loader, val_loader)
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
