import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.launch import bootstrap, load_config

bootstrap()

from models.medvae import MVAE
from datasets.stage_1_dataset import build_stage_1_dataset
from utils.stage_1_trainer import TrainerStage1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="jepa_adaptation/configs/stage_1.yaml")
    parser.add_argument("--model-config", default="jepa_adaptation/configs/model.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, args.model_config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    mvae = MVAE(
        ddconfig=model_cfg["ddconfig"],
        embed_dim=int(model_cfg.get("embed_dim", 4)),
        spatial_dims=int(model_cfg.get("spatial_dims", 2)),
        apply_channel_ds=bool(model_cfg.get("apply_channel_ds", True)),
        medvae_pretrained=model_cfg.get("medvae_pretrained"),
        ckpt_path=model_cfg.get("ckpt_path"),
    )

    common = dict(
        root=data_cfg.get("root", "~/.medmnist"),
        size=int(data_cfg.get("size", 224)),
        as_rgb=bool(data_cfg.get("as_rgb", False)),
        download=bool(data_cfg.get("download", True)),
        fraction=float(data_cfg.get("fraction", 1.0)),
        seed=int(data_cfg.get("seed", 42)),
    )
    train_ds = build_stage_1_dataset(split="train", **common)
    val_ds = build_stage_1_dataset(split="val", **common)

    TrainerStage1(mvae, cfg, train_ds, val_ds).fit()


if __name__ == "__main__":
    main()
