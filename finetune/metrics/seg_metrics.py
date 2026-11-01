import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex


class SegMetrics(nn.Module):
    def __init__(
        self,
        num_classes: int = 26,
        ignore_index: int = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.num_classes = num_classes

        # Dice score = F1 score en segmentation
        # average="none": one score per class
        # average="macro": mean over all classes
        self.dice_per_class = MulticlassF1Score(
            num_classes=num_classes,
            average="none",          # returns a vector of size num_classes
            ignore_index=ignore_index,
        ).to(device)

        self.dice_mean = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",         # unweighted mean over all classes
            ignore_index=ignore_index,
        ).to(device)

        # IoU computed like the Dice (same accumulation over the dataset, same macro
        # mean). The torchmetrics MeanIoU expects one-hot masks by default and returned
        # wrong values on the class-index masks used here.
        self.iou_mean = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
            ignore_index=ignore_index,
        ).to(device)

        # For the means over the arteries only (background excluded, ARCADE standard)
        self.iou_per_class = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="none",
            ignore_index=ignore_index,
        ).to(device)
        # Predicted / true pixels per class: a class absent from both sides
        # is excluded from the means (as in the torchmetrics macro mean)
        self.register_buffer("class_pixels", torch.zeros(num_classes, dtype=torch.long, device=device))

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        # Convert the logits to predicted classes (argmax over the class dimension)
        preds = logits.argmax(dim=1)   # (B, H, W)

        self.dice_per_class.update(preds, targets)
        self.dice_mean.update(preds, targets)
        self.iou_mean.update(preds, targets)
        self.iou_per_class.update(preds, targets)
        self.class_pixels += torch.bincount(preds.flatten(), minlength=self.num_classes)[:self.num_classes]
        self.class_pixels += torch.bincount(targets.flatten(), minlength=self.num_classes)[:self.num_classes]

    def compute(self) -> dict:
        dice_per_class = self.dice_per_class.compute()  # tensor (num_classes,)
        dice_mean      = self.dice_mean.compute()        # scalar tensor
        iou_mean       = self.iou_mean.compute()         # scalar tensor

        iou_per_class  = self.iou_per_class.compute()
        present_fg     = self.class_pixels[1:] > 0     # arteries present (predicted or true)

        results = {
            "dice_mean":      dice_mean.item(),
            "iou_mean":       iou_mean.item(),
            # Means over the arteries only (classes 1..N-1 present), background excluded
            "dice_fg_mean":   dice_per_class[1:][present_fg].mean().item() if present_fg.any() else 0.0,
            "iou_fg_mean":    iou_per_class[1:][present_fg].mean().item() if present_fg.any() else 0.0,
            "dice_per_class": dice_per_class.tolist(),
            "iou_per_class":  iou_per_class.tolist(),
        }

        # Add each class separately for wandb logging
        for i, score in enumerate(dice_per_class.tolist()):
            results[f"dice_class_{i}"] = score

        return results

    def reset(self) -> None:
        """Reset the accumulators — to call at the start of each epoch."""
        self.dice_per_class.reset()
        self.dice_mean.reset()
        self.iou_mean.reset()
        self.iou_per_class.reset()
        self.class_pixels.zero_()


def build_metrics(config: dict, device: torch.device) -> SegMetrics:
    return SegMetrics(
        num_classes=config["num_classes"],
        ignore_index=config.get("ignore_index", None),
        device=device,
    )


if __name__ == "__main__":
    # Test rapide
    NUM_CLASSES = 26
    B, H, W = 2, 256, 256
    device = torch.device("cpu")

    metrics = SegMetrics(num_classes=NUM_CLASSES, device=device)

    # Simule 3 batches
    for _ in range(3):
        logits  = torch.randn(B, NUM_CLASSES, H, W)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        metrics.update(logits, targets)

    results = metrics.compute()

    print(f"Dice mean : {results['dice_mean']:.4f}")
    print(f"IoU mean  : {results['iou_mean']:.4f}")
    print(f"Dice per class (first 5): "
          f"{[round(v, 4) for v in results['dice_per_class'][:5]]}")
    print("SegMetrics OK")
