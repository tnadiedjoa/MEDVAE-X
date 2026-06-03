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


def build_seg_head(config: dict) -> SegHead:
    return SegHead(
        in_channels=config.get("latent_channels", 1),
        num_classes=config.get("num_classes", 26),
        base_channels=config.get("base_channels", 64),
        n_upsample=config.get("n_upsample", 4),
    )