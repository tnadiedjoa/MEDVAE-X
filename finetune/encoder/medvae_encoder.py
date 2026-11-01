import torch
import torch.nn as nn
from medvae import MVAE


class MedVAEEncoder(nn.Module):
    """
    Frozen MedVAE encoder used as a feature extractor.

    Condition B: HuggingFace pre-trained weights (checkpoint_path=None).
    Condition C: weights fine-tuned on ARCADE, loaded from checkpoint_path.

    In both cases the encoder is fully frozen after loading.
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

        print(f"Loading MedVAE ({model_name})...")
        self.mvae = MVAE(
            model_name=model_name,
            modality=modality,
        )
        self.mvae = self.mvae.to(device)

        if checkpoint_path is not None:
            self._load_finetuned(checkpoint_path, device)

        # Freeze all weights
        for param in self.mvae.parameters():
            param.requires_grad = False
        self.mvae.eval()

        source = f"checkpoint ({checkpoint_path})" if checkpoint_path else "HuggingFace pre-trained"
        print(f"MedVAE loaded and frozen — source: {source}")

    def _load_finetuned(self, checkpoint_path: str, device: torch.device) -> None:
        print(f"Loading the fine-tuned checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Supports PyTorch Lightning, Phase1Trainer and plain PyTorch checkpoints
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "autoencoder" in ckpt:   # format Phase1Trainer (jepa-adaptation)
                state = ckpt["autoencoder"]
            elif "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
        else:
            raise ValueError(f"Format de checkpoint non reconnu : {type(ckpt)}")

        # Try to load the encoder weights only
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }

        if encoder_state and hasattr(self.mvae, "encoder"):
            result = self.mvae.encoder.load_state_dict(encoder_state, strict=False)
            print(f"  Encoder weights loaded — missing: {result.missing_keys[:5]}")
        else:
            # Fallback: load the whole model with strict=False
            result = self.mvae.load_state_dict(state, strict=False)
            print(f"  Full model weights loaded (strict=False) — missing: {result.missing_keys[:5]}")

    def train(self, mode: bool = True):
        # The encoder always stays in eval mode, so that its normalisation layers behave as at inference
        super().train(mode)
        self.mvae.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into its latent representation.

        MedVAE expects images in [-1, 1].
        The dataset produces images in [0, 1].
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
    Frozen MedVAE used as a preprocessor: encodes then decodes the image.

    Condition D: the 512×512 image goes through the (frozen) encode→decode cycle
    before being given to the U-Net. The U-Net therefore sees images at the
    MedVAE reconstruction quality and adapts to it during training.

    Output: reconstructed 512×512 image in [0, 1], same format as the
    original image — the U-Net sees no difference of interface.
    """

    def __init__(
        self,
        model_name: str = "medvae_4_1_2d",
        modality: str = "xray",
        device: torch.device = torch.device("cpu"),
        chunk_size: int = 2,
    ):
        super().__init__()

        # Images processed in groups: MedVAE attention at 512×512 costs ~1 GB per
        # image, and a batch of 8 at once exceeds the 24 GB of a 3090.
        # No effect on the result (frozen model, per-image normalisations).
        self.chunk_size = chunk_size

        print(f"Loading the MedVAE autoencoder ({model_name})...")
        self.mvae = MVAE(model_name=model_name, modality=modality).to(device)

        for param in self.mvae.parameters():
            param.requires_grad = False
        self.mvae.eval()

        print("MedVAE autoencoder loaded and frozen.")

    def train(self, mode: bool = True):
        super().train(mode)
        self.mvae.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 512, 512] in [0, 1]
        output: [B, 1, 512, 512] in [0, 1]  (reconstructed by MedVAE)

        Cycle: [0,1] → [-1,1] → encode → decode → [-1,1] → [0,1]
        """
        x_norm = x * 2.0 - 1.0

        with torch.no_grad():
            reconstructed = torch.cat([
                self.mvae.decode(self.mvae.encode(chunk))  # [b,1,128,128] → [b,1,512,512] in [-1,1]
                for chunk in x_norm.split(self.chunk_size)
            ])

        # Map [-1,1] → [0,1] and clip for safety
        return ((reconstructed + 1.0) / 2.0).clamp(0.0, 1.0)
