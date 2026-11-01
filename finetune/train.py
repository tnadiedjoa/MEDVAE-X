import argparse
import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from finetune.config import arcade_root, load_config
from finetune.dataset import ArcadeDataset, seed_worker_augmentations, split_dataset
from finetune.models import build_unet, build_seg_head
from finetune.runs import add_run_args, apply_overrides, create_run
from finetune.trainer import Trainer


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
        print("CPU only")
    return device


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]

    noise = data_cfg.get("gauss_noise_std_range")   # E07; None = original behaviour
    degradation = data_cfg.get("degradation_aug_p")  # E15; None = no degradation

    if data_cfg.get("validation", "split") == "official":
        # E08: all of seg_train (1000 images) for training, official seg_val
        # (200 images) for model selection
        train_dataset = ArcadeDataset(data_cfg["train_images"], data_cfg["train_ann"],
                                      augment=True, noise_std_range=noise,
                                      degradation_p=degradation)
        seg_val = arcade_root() / "dataset_phase_1" / "segmentation_dataset" / "seg_val"
        val_dataset   = ArcadeDataset(
            data_cfg.get("official_val_images", str(seg_val / "images")),
            data_cfg.get("official_val_ann", str(seg_val / "annotations" / "seg_val.json")),
            augment=False,
        )
    else:
        # Original behaviour: seg_train split 80/20 according to the seed
        train_ids, val_ids = split_dataset(
            data_cfg["train_ann"],
            train_ratio=data_cfg["train_ratio"],
            seed=config["experiment"]["seed"],
        )
        train_dataset = ArcadeDataset(data_cfg["train_images"], data_cfg["train_ann"],
                                      train_ids, augment=True, noise_std_range=noise,
                                      degradation_p=degradation)
        val_dataset   = ArcadeDataset(data_cfg["train_images"], data_cfg["train_ann"],
                                      val_ids,   augment=False)
    test_dataset  = ArcadeDataset(data_cfg["val_images"],   data_cfg["val_ann"],
                                  augment=False)

    print(f"Train : {len(train_dataset)} images")
    print(f"Val   : {len(val_dataset)} images")
    print(f"Test  : {len(test_dataset)} images")

    loader_kwargs = dict(
        batch_size     = config["training"]["batch_size"],
        num_workers    = data_cfg["num_workers"],
        pin_memory     = data_cfg["pin_memory"],
        worker_init_fn = seed_worker_augmentations,   # augmentations specific to each worker and epoch
    )

    return (
        DataLoader(train_dataset, shuffle=True,  **loader_kwargs),
        DataLoader(val_dataset,   shuffle=False, **loader_kwargs),
        DataLoader(test_dataset,  shuffle=False, **loader_kwargs),
    )


def build_model(config: dict, device: torch.device):
    """
    Build the model of the condition:
    - Condition A: full U-Net
    - Condition B/C: frozen MedVAE encoder + segmentation head
    """
    condition = config["experiment"]["condition"]

    if condition == "A":
        model = build_unet(config["model"])

    elif condition in ("B", "C"):
        from finetune.encoder import MedVAEEncoder
        import torch.nn as nn

        # Sans checkpoint, C tournerait silencieusement comme B
        if condition == "C" and not config["encoder"].get("checkpoint_path"):
            raise ValueError("Condition C: encoder.checkpoint_path missing "
                             "(--set encoder.checkpoint_path=<fine-tuned MedVAE>)")

        encoder = MedVAEEncoder(
            model_name=config["encoder"]["model_name"],
            modality=config["encoder"]["modality"],
            device=device,
            checkpoint_path=config["encoder"].get("checkpoint_path", None),
        )
        seg_head = build_seg_head(config["model"])

        class EncoderWithHead(nn.Module):
            def __init__(self, enc, head):
                super().__init__()
                self.encoder  = enc
                self.seg_head = head

            def forward(self, x):
                latent = self.encoder.encode(x)
                return self.seg_head(latent)

        model = EncoderWithHead(encoder, seg_head)

    elif condition == "D":
        from finetune.encoder import MedVAEAutoencoder
        import torch.nn as nn

        # Frozen MedVAE: preprocesses the images (encode→decode) before the U-Net
        autoencoder = MedVAEAutoencoder(
            model_name=config["encoder"]["model_name"],
            modality=config["encoder"]["modality"],
            device=device,
        )
        unet = build_unet(config["model"])

        class AutoencoderUNet(nn.Module):
            """
            Preprocess each image with MedVAE (encode→decode), then segment
            it with a trainable U-Net. The U-Net thus adapts to the MedVAE
            reconstruction quality during training.
            """
            def __init__(self, ae, unet):
                super().__init__()
                self.autoencoder = ae
                self.unet        = unet

            def forward(self, x):
                reconstructed = self.autoencoder(x)  # [B,1,512,512] in [0,1]
                return self.unet(reconstructed)

        model = AutoencoderUNet(autoencoder, unet)

    elif condition == "R":
        import torch.nn as nn
        import torch.nn.functional as F

        # Control (E16): same head as B and C, on the image reduced to the latent size
        # (128×128×1, as many values as the MedVAE latent) instead of the latent
        seg_head = build_seg_head(config["model"])

        class DownsampledImageHead(nn.Module):
            def __init__(self, head, factor):
                super().__init__()
                self.seg_head, self.factor = head, factor

            def forward(self, x):
                small = F.avg_pool2d(x, self.factor) * 2 - 1   # [0,1] → [-1,1], like the MedVAE input
                return self.seg_head(small)

        model = DownsampledImageHead(seg_head, factor=2 ** config["model"].get("n_upsample", 2))

    else:
        raise ValueError(f"Condition inconnue : {condition}")

    # Count the trainable parameters only
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    add_run_args(parser)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    print(f"Experiment: {config['experiment']['name']}")
    run_name = args.run_name or config["experiment"]["name"]
    config["logging"]["save_dir"] = create_run(config, run_name, args.overrides)

    set_seed(config["experiment"]["seed"])
    device = get_device()

    train_loader, val_loader, test_loader = build_dataloaders(config)
    model   = build_model(config, device)
    trainer = Trainer(model=model, config=config, device=device)
    trainer.fit(train_loader, val_loader)
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
