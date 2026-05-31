import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

import torch.nn.functional as F

def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=(1,1), padding=(1,0))
    p2 = -F.max_pool2d(-p1,  kernel_size=(1,3), stride=(1,1), padding=(0,1))
    return p2

def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=(3,3), stride=(1,1), padding=(1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters=5):
    skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img   = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel  = skel + F.relu(delta - skel * delta)
    return skel

def cl_dice_loss(pred, target, iters=5, smooth=1.0):
    skel_pred   = soft_skel(pred,   iters)
    skel_target = soft_skel(target, iters)
    tprec = (torch.sum(skel_pred * target)   + smooth) / (torch.sum(skel_pred)   + smooth)
    tsens = (torch.sum(skel_target * pred)   + smooth) / (torch.sum(skel_target) + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)

class SegLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 26,
        dice_weight: float = 0.4,
        ce_weight: float = 0.4,
        cl_weight: float = 0.2,
        ignore_index: int = None,
        skel_iters: int = 5,
        
    ):
        super().__init__()

        self.cl_weight   = cl_weight
        self.num_classes = num_classes
        self.skel_iters  = skel_iters

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
        # dice + ce  sur logits vs targets
        dice = self.dice_loss(logits, targets)
        ce   = self.ce_loss(logits, targets)

        # cl dice sur les probabilités vs target one-hot
        probs     = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(targets, self.num_classes).permute(0,3,1,2).float()
        cl        = cl_dice_loss(probs, target_oh, self.skel_iters)

        total = self.dice_weight * dice + self.ce_weight * ce + self.cl_weight * cl

        loss_dict = {
            "loss":      total.item(),
            "dice_loss": dice.item(),
            "ce_loss":   ce.item(),
            "cl_loss":   cl.item(),
        }

        return total, loss_dict


def build_loss(config: dict) -> SegLoss:
    return SegLoss(
        dice_weight=config.get("dice_weight", 0.5),
        ce_weight=config.get("ce_weight", 0.5),
        ignore_index=config.get("ignore_index", None),
        cl_weight=config.get("cl_weight", 0.2),
        skel_iters=config.get("skel_iters", 5),
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
