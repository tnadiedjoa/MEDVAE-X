import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net with a pre-trained segmentation_models_pytorch backbone."""
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 1,
        num_classes: int = 26,
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # no softmax (it is in the loss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_unet(config: dict) -> UNet:
    return UNet(
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
    )


if __name__ == "__main__":
    # Test rapide
    model = UNet(
        encoder_name="resnet34",
        encoder_weights=None,   
        in_channels=1,
        num_classes=26,
    )

    # Simulate a batch of 2 greyscale 512x512 images
    dummy_input = torch.randn(2, 1, 512, 512)
    output = model(dummy_input)

    print(f"Input  : {dummy_input.shape}")   # (2, 1, 512, 512)
    print(f"Output : {output.shape}")        # (2, 26, 512, 512)

    # Check that the predictions have the right shape
    preds = output.argmax(dim=1)
    print(f"Preds  : {preds.shape}")         # (2, 512, 512)

    print("UNet OK")
