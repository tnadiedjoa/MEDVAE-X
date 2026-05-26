import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    Wrapper autour du U-Net de SMP.

    Args:
        encoder_name    : backbone de l'encodeur, ex. "resnet34", "resnet50",
                          "efficientnet-b4". Voir smp.encoders.get_encoder_names().
        encoder_weights : poids initiaux du backbone. "imagenet" pour du transfer
                          learning, None pour initialisation aléatoire.
        in_channels     : nombre de canaux en entrée. 1 pour les images en
                          niveaux de gris (ARCADE).
        num_classes     : nombre de classes de sortie. 26 pour ARCADE.
    """

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
            activation=None,  # pas de softmax (dans la loss)
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

    # Simule un batch de 2 images 512x512 en niveaux de gris
    dummy_input = torch.randn(2, 1, 512, 512)
    output = model(dummy_input)

    print(f"Input  : {dummy_input.shape}")   # (2, 1, 512, 512)
    print(f"Output : {output.shape}")        # (2, 26, 512, 512)

    # Vérifie que les prédictions ont la bonne forme
    preds = output.argmax(dim=1)
    print(f"Preds  : {preds.shape}")         # (2, 512, 512)

    print("UNet OK")
