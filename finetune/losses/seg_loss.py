import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class SegLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 26,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_index: int = None,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        # Dice Loss de SMP mode MULTICLASS pour 26 classes
        # from_logits=True car notre U-Net retourne des logits bruts
        self.dice_loss = DiceLoss(
            mode="multiclass",
            classes=list(range(num_classes)),
            from_logits=True,
            smooth=1e-6,
        )

        # Cross-Entropy standard de PyTorch
        # ignore_index permet d'ignorer le fond si besoin
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        dice = self.dice_loss(logits, targets)
        ce   = self.ce_loss(logits, targets)

        total = self.dice_weight * dice + self.ce_weight * ce

        loss_dict = {
            "loss":      total.item(),
            "dice_loss": dice.item(),
            "ce_loss":   ce.item(),
        }

        return total, loss_dict


def build_loss(config: dict) -> SegLoss:
    return SegLoss(
        dice_weight=config.get("dice_weight", 0.5),
        ce_weight=config.get("ce_weight", 0.5),
        ignore_index=config.get("ignore_index", None),
    )


if __name__ == "__main__":
    import torch

    NUM_CLASSES = 26
    B, H, W = 2, 256, 256

    criterion = SegLoss(num_classes=NUM_CLASSES, dice_weight=0.5, ce_weight=0.5)

    # Simule des logits et un masque ground truth
    logits  = torch.randn(B, NUM_CLASSES, H, W)
    targets = torch.randint(0, NUM_CLASSES, (B, H, W))

    total_loss, loss_dict = criterion(logits, targets)

    print(f"Total loss : {loss_dict['loss']:.4f}")
    print(f"Dice loss  : {loss_dict['dice_loss']:.4f}")
    print(f"CE loss    : {loss_dict['ce_loss']:.4f}")
    print("SegLoss OK")
