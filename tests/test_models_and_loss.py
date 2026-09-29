"""Têtes de segmentation et loss."""

import pytest
import torch
import torch.nn.functional as F

from finetune.losses import SegLoss
from finetune.models import LatentUNetHead, SegHead, build_seg_head

NUM_CLASSES = 26


@pytest.mark.parametrize("architecture, cls", [("seg_head", SegHead),
                                               ("latent_unet_head", LatentUNetHead)])
def test_heads_upsample_latent_to_image(architecture, cls):
    head = build_seg_head({"architecture": architecture, "latent_channels": 1,
                           "num_classes": NUM_CLASSES, "base_channels": 32, "n_upsample": 2})
    assert isinstance(head, cls)
    out = head(torch.randn(2, 1, 32, 32))            # latent f=4 → image ×4
    assert out.shape == (2, NUM_CLASSES, 128, 128)


def test_unknown_head_architecture_fails():
    with pytest.raises(ValueError):
        build_seg_head({"architecture": "inconnue"})


def test_loss_prefers_correct_prediction():
    torch.manual_seed(0)
    targets = torch.zeros(2, 64, 64, dtype=torch.long)
    targets[:, 20:30, :] = 5
    criterion = SegLoss(num_classes=NUM_CLASSES, dice_weight=0.5, ce_weight=0.5, cl_weight=0.0)

    perfect = F.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float() * 20
    random = torch.randn(2, NUM_CLASSES, 64, 64)
    loss_perfect, parts = criterion(perfect, targets)
    loss_random, _ = criterion(random, targets)

    assert loss_perfect < loss_random
    assert {"loss", "dice_loss", "ce_loss", "cl_loss"} <= parts.keys()


def test_condition_r_segments_downsampled_image():
    """Témoin E16 : même tête que B/C, sur l'image réduite à 128×128."""
    import torch

    from finetune.config import REPO_ROOT, load_config
    from finetune.train import build_model

    config = load_config(str(REPO_ROOT / "finetune" / "configs" / "condition_r.yaml"))
    config["model"]["base_channels"] = 8
    model = build_model(config, torch.device("cpu"))
    assert model(torch.rand(1, 1, 512, 512)).shape == (1, 26, 512, 512)
