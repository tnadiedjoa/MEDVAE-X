import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.launch import bootstrap, load_config

bootstrap()

from models.medvae import MVAE
from models.mvae_jepa import MVAE_JEPA
from datasets.stage_2_dataset import build_stage_2_dataset
from datasets.arcade_dataset import get_pretraining_datasets
from utils.stage_2_trainer import TrainerStage2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="jepa_adaptation/configs/stage_2.yaml")
    parser.add_argument("--model-config", default="jepa_adaptation/configs/model.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.model_config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    jepa_cfg = cfg.get("jepa", {})
    train_cfg = cfg.get("training", {})

    mvae = MVAE(
        ddconfig=model_cfg["ddconfig"],
        embed_dim=int(model_cfg.get("embed_dim", 4)),
        spatial_dims=int(model_cfg.get("spatial_dims", 2)),
        apply_channel_ds=bool(model_cfg.get("apply_channel_ds", True)),
    )

    stage1_ckpt = train_cfg.get("stage1_ckpt")
    if stage1_ckpt:
        mvae.load_weights(stage1_ckpt)

    model = MVAE_JEPA(
        mvae,
        patch_size=int(jepa_cfg.get("patch_size", 16)),
        target_mask_ratio=float(jepa_cfg.get("target_mask_ratio", 0.4)),
        predictor_hidden_dim=jepa_cfg.get("predictor_hidden_dim"),
        predictor_depth=int(jepa_cfg.get("predictor_depth", 3)),
        ema_momentum=float(jepa_cfg.get("ema_momentum", 0.996)),
        sample_posterior=bool(jepa_cfg.get("sample_posterior", False)),
        freeze_decoder=bool(jepa_cfg.get("freeze_decoder", True)),
        freeze_encoder=bool(jepa_cfg.get("freeze_encoder", False)),
        use_penultimate=bool(jepa_cfg.get("use_penultimate", True)),
    )

    source = data_cfg.get("source", "medmnist")
    if source == "arcade":
        train_ds, val_ds = get_pretraining_datasets(
            img_size=int(data_cfg.get("img_size", 512)),
            val_ratio=float(data_cfg.get("val_ratio", 0.1)),
            seed=int(data_cfg.get("seed", 42)),
            augment=bool(data_cfg.get("augment", True)),
        )
    else:
        common = dict(
            root=data_cfg.get("root", "~/.medmnist"),
            size=int(data_cfg.get("size", 224)),
            as_rgb=bool(data_cfg.get("as_rgb", False)),
            download=bool(data_cfg.get("download", True)),
            fraction=float(data_cfg.get("fraction", 1.0)),
            seed=int(data_cfg.get("seed", 42)),
        )
        train_ds = build_stage_2_dataset(split="train", **common)
        val_ds   = build_stage_2_dataset(split="val",   **common)

    TrainerStage2(model, cfg, train_ds, val_ds).fit()


if __name__ == "__main__":
    main()
