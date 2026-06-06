import torch
import torch.nn as nn
from medvae import MVAE


class MedVAEEncoder(nn.Module):
    """
    Encodeur MedVAE gelé — utilisé comme feature extractor.

    Le modèle est chargé depuis HuggingFace (stanfordmimi/MedVAE)
    au moment de l'instanciation. Tous les poids sont gelés —
    MedVAE joue uniquement le rôle de compresseur, il n'est pas
    modifié par la loss de segmentation.
    """

    def __init__(
        self,
        model_name: str = "medvae_4_1_2d",
        modality: str = "xray",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.model_name = model_name
        self.modality   = modality

        # Charge le modèle MedVAE depuis HuggingFace
        print(f"Chargement de MedVAE ({model_name})...")
        self.mvae = MVAE(
            model_name=model_name,
            modality=modality,
        )
        self.mvae = self.mvae.to(device)

        # Gèle tous les poids — MedVAE ne sera pas entraîné
        for param in self.mvae.parameters():
            param.requires_grad = False
        self.mvae.eval()

        print(f"MedVAE chargé et gelé.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode une image en représentation latente.

        MedVAE attend des images normalisées entre [-1, 1].
        Notre dataset produit des images dans [0, 1], donc on
        recentre ici.
        """
        # Renormalise [0,1] → [-1,1]
        x_norm = x * 2.0 - 1.0

        with torch.no_grad():
            latent = self.mvae.encode(x_norm)

        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias de encode() pour compatibilité avec nn.Module."""
        return self.encode(x)


def build_encoder(config: dict, device: torch.device) -> MedVAEEncoder:
    """
    Construit l'encodeur MedVAE depuis la section 'encoder' du yaml.
    """
    return MedVAEEncoder(
        model_name=config.get("model_name", "medvae_4_1_2d"),
        modality=config.get("modality", "xray"),
        device=device,
    )