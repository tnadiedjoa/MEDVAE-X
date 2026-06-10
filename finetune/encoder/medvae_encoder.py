import torch
import torch.nn as nn
from medvae import MVAE


class MedVAEEncoder(nn.Module):
    """
    Encodeur MedVAE gelé — utilisé comme feature extractor.

    Condition B : poids pré-entraînés HuggingFace (checkpoint_path=None).
    Condition C : poids fine-tunés sur ARCADE chargés depuis checkpoint_path.

    Dans les deux cas l'encodeur est entièrement gelé après chargement.
    """

    def __init__(
        self,
        model_name: str = "medvae_4_1_2d",
        modality: str = "xray",
        device: torch.device = torch.device("cpu"),
        checkpoint_path: str = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.modality   = modality

        print(f"Chargement de MedVAE ({model_name})...")
        self.mvae = MVAE(
            model_name=model_name,
            modality=modality,
        )
        self.mvae = self.mvae.to(device)

        if checkpoint_path is not None:
            self._load_finetuned(checkpoint_path, device)

        # Gèle tous les poids
        for param in self.mvae.parameters():
            param.requires_grad = False
        self.mvae.eval()

        source = f"checkpoint ({checkpoint_path})" if checkpoint_path else "HuggingFace pré-entraîné"
        print(f"MedVAE chargé et gelé — source : {source}")

    def _load_finetuned(self, checkpoint_path: str, device: torch.device) -> None:
        print(f"Chargement du checkpoint fine-tuné : {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Supporte les checkpoints PyTorch Lightning et PyTorch natifs
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            raise ValueError(f"Format de checkpoint non reconnu : {type(ckpt)}")

        # Essaie de charger uniquement les poids de l'encodeur
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }

        if encoder_state and hasattr(self.mvae, "encoder"):
            result = self.mvae.encoder.load_state_dict(encoder_state, strict=False)
            print(f"  Poids encodeur chargés — manquants : {result.missing_keys[:5]}")
        else:
            # Fallback : charge tout le modèle avec strict=False
            result = self.mvae.load_state_dict(state, strict=False)
            print(f"  Poids modèle complet chargés (strict=False) — manquants : {result.missing_keys[:5]}")

    def train(self, mode: bool = True):
        # L'encodeur reste toujours en eval pour que ses BN utilisent les stats fixes
        super().train(mode)
        self.mvae.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode une image en représentation latente.

        MedVAE attend des images dans [-1, 1].
        Notre dataset produit des images dans [0, 1].
        """
        x_norm = x * 2.0 - 1.0

        with torch.no_grad():
            latent = self.mvae.encode(x_norm)

        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def build_encoder(config: dict, device: torch.device) -> MedVAEEncoder:
    return MedVAEEncoder(
        model_name=config.get("model_name", "medvae_4_1_2d"),
        modality=config.get("modality", "xray"),
        device=device,
        checkpoint_path=config.get("checkpoint_path", None),
    )


class MedVAEAutoencoder(nn.Module):
    """
    MedVAE gelé utilisé comme préprocesseur : encode puis décode l'image.

    Condition D : l'image 512×512 passe par le cycle encode→decode (gelé)
    avant d'être donnée au U-Net. Le U-Net voit donc des images à la
    qualité de reconstruction MedVAE et s'y adapte pendant l'entraînement.

    Sortie : image reconstruite 512×512 dans [0, 1], même format que
    l'image originale — le U-Net ne voit aucune différence d'interface.
    """

    def __init__(
        self,
        model_name: str = "medvae_4_1_2d",
        modality: str = "xray",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        print(f"Chargement de MedVAE autoencoder ({model_name})...")
        self.mvae = MVAE(model_name=model_name, modality=modality).to(device)

        for param in self.mvae.parameters():
            param.requires_grad = False
        self.mvae.eval()

        print("MedVAE autoencoder chargé et gelé.")

    def train(self, mode: bool = True):
        super().train(mode)
        self.mvae.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, 512, 512] dans [0, 1]
        sortie : [B, 1, 512, 512] dans [0, 1]  (reconstruite par MedVAE)

        Cycle : [0,1] → [-1,1] → encode → decode → [-1,1] → [0,1]
        """
        x_norm = x * 2.0 - 1.0

        with torch.no_grad():
            latent        = self.mvae.encode(x_norm)       # [B,1,128,128]
            reconstructed = self.mvae.decode(latent)       # [B,1,512,512] dans [-1,1]

        # Renormalise [-1,1] → [0,1] et clippe pour sécurité
        return ((reconstructed + 1.0) / 2.0).clamp(0.0, 1.0)
