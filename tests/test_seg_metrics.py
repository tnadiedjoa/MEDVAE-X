"""Vérifie SegMetrics contre un calcul manuel du Dice et de l'IoU."""

import pytest
import torch
import torch.nn.functional as F

from finetune.metrics import SegMetrics

NUM_CLASSES = 26


def manual_scores(preds, targets):
    """Dice et IoU moyens sur les classes présentes (dans la prédiction ou la cible)."""
    dices, ious = [], []
    for c in range(NUM_CLASSES):
        p, t = preds == c, targets == c
        inter = (p & t).sum().item()
        total = p.sum().item() + t.sum().item()
        if total:
            dices.append(2 * inter / total)
            ious.append(inter / (total - inter))
    return sum(dices) / len(dices), sum(ious) / len(ious)


def make_targets():
    t = torch.zeros(2, 64, 64, dtype=torch.long)
    t[:, 10:20, 10:40] = 3
    t[:, 30:35, :] = 7
    return t


def make_cases():
    t = make_targets()
    g = torch.Generator().manual_seed(0)
    noise = torch.rand(t.shape, generator=g) < 0.3
    return {
        "parfait": t.clone(),
        "tout_fond": torch.zeros_like(t),
        "bruit": torch.where(noise, torch.randint(0, NUM_CLASSES, t.shape, generator=g), t),
        "classe_fantome": torch.where(t == 7, torch.full_like(t, 12), t),
    }


@pytest.mark.parametrize("name", list(make_cases()))
def test_matches_manual_computation(name):
    targets = make_targets()
    preds = make_cases()[name]
    logits = F.one_hot(preds, NUM_CLASSES).permute(0, 3, 1, 2).float()

    metrics = SegMetrics(num_classes=NUM_CLASSES)
    # Deux batchs : vérifie l'accumulation sur le dataset
    for k in range(2):
        metrics.update(logits[k:k + 1], targets[k:k + 1])
    results = metrics.compute()

    dice, iou = manual_scores(preds, targets)
    assert results["dice_mean"] == pytest.approx(dice, abs=1e-4)
    assert results["iou_mean"] == pytest.approx(iou, abs=1e-4)
    assert results["dice_mean"] >= results["iou_mean"] - 1e-6
