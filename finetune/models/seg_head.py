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

        # Initial projection: adapts the latent channels
        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        # Blocs d'upsampling
        self.up_blocks = nn.ModuleList()
        ch = base_channels
        for i in range(n_upsample):
            out_ch = max(ch // 2, 32)   # fewer channels at each level
            self.up_blocks.append(ConvBlock(ch, out_ch))
            ch = out_ch

        # Final classification head
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)

        return self.head(x)


class LatentUNetHead(nn.Module):
    """
    Head with a small U-Net at the latent resolution (E04).

    The 128×128 latent is reduced down to 16×16 (global context, needed to tell
    the artery segments apart) then rebuilt with skip connections (fine
    details), before the upsampling to 512×512 as in SegHead.
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

        # Encoder: resolution / 2 and channels × 2 at each level
        chans = [base_channels]
        self.down_blocks = nn.ModuleList()
        for _ in range(depth):
            out_ch = min(chans[-1] * 2, max_channels)
            self.down_blocks.append(ConvBlock(chans[-1], out_ch))
            chans.append(out_ch)

        # Decoder: back up to the latent resolution with the skips
        self.up_blocks = nn.ModuleList()
        ch = chans[-1]
        for skip_ch in reversed(chans[:-1]):
            self.up_blocks.append(ConvBlock(ch + skip_ch, skip_ch))
            ch = skip_ch

        # Upsampling to the image resolution (latent f=4 → ×2 ×2)
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
        skips.pop()  # the lowest level has no skip

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
    raise ValueError(f"Unknown head architecture: {architecture}")