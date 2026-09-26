import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Bloc convolutif standard : Conv → BN → ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 26,
        base_channels: int = 64,
        n_upsample: int = 4,
    ):
        super().__init__()

        # Projection initiale : adapte les canaux latents
        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        # Blocs d'upsampling
        self.up_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(n_upsample):
            out_ch = max(ch // 2, 32)   # diminue les canaux à chaque niveau
            self.up_blocks.append(ConvBlock(ch, out_ch))
            ch = out_ch

        # Tête de classification finale
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)

        return self.head(x)


class LatentUNetHead(nn.Module):
    """
    Tête avec un petit U-Net à la résolution du latent (E04).

    Le latent 128×128 est réduit jusqu'à 16×16 (contexte global, nécessaire pour
    distinguer les segments d'artères) puis reconstruit avec des connexions skip
    (détails fins), avant l'upsampling vers 512×512 comme dans SegHead.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 26,
        base_channels: int = 64,
        depth: int = 3,
        n_upsample: int = 2,
        max_channels: int = 512,
    ):
        super().__init__()

        self.stem = ConvBlock(in_channels, base_channels)

        # Encodeur : résolution / 2 et canaux × 2 à chaque niveau
        chans = [base_channels]
        self.down_blocks = nn.ModuleList()
        for _ in range(depth):
            out_ch = min(chans[-1] * 2, max_channels)
            self.down_blocks.append(ConvBlock(chans[-1], out_ch))
            chans.append(out_ch)

        # Décodeur : remonte jusqu'à la résolution du latent avec les skips
        self.up_blocks = nn.ModuleList()
        ch = chans[-1]
        for skip_ch in reversed(chans[:-1]):
            self.up_blocks.append(ConvBlock(ch + skip_ch, skip_ch))
            ch = skip_ch

        # Upsampling vers la résolution de l'image (latent f=4 → ×2 ×2)
        self.out_blocks = nn.ModuleList()
        for _ in range(n_upsample):
            out_ch = max(ch // 2, 32)
            self.out_blocks.append(ConvBlock(ch, out_ch))
            ch = out_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = [x]
        for block in self.down_blocks:
            x = block(F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()  # le niveau le plus bas n'a pas de skip

        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skips.pop()], dim=1))

        for block in self.out_blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)

        return self.head(x)


def build_seg_head(config: dict) -> nn.Module:
    architecture = config.get("architecture", "seg_head")
    if architecture == "seg_head":
        return SegHead(
            in_channels=config.get("latent_channels", 1),
            num_classes=config.get("num_classes", 26),
            base_channels=config.get("base_channels", 64),
            n_upsample=config.get("n_upsample", 4),
        )
    if architecture == "latent_unet_head":
        return LatentUNetHead(
            in_channels=config.get("latent_channels", 1),
            num_classes=config.get("num_classes", 26),
            base_channels=config.get("base_channels", 64),
            depth=config.get("depth", 3),
            n_upsample=config.get("n_upsample", 2),
        )
    raise ValueError(f"Architecture de tête inconnue : {architecture}")