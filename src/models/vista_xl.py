
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from xlstm.blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig


class VISTAXLConvBlock(nn.Module):
    """Two convolution layers with BatchNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VISTAXL(nn.Module):
    """
    Swin-Tiny encoder + mLSTM + U-Net-style decoder.

    Input:  (B, 3, H, W)
    Output: (B, 1, H, W) segmentation logits
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        mlstm_config = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                embedding_dim=768,
                num_heads=4,
                context_length=49,
                dropout=0.0,
            ),
            _num_blocks=1,
            _block_idx=0,
        )
        self.mlstm = mLSTMBlock(mlstm_config)

        self.decode3 = VISTAXLConvBlock(768 + 384, 256)
        self.decode2 = VISTAXLConvBlock(256 + 192, 128)
        self.decode1 = VISTAXLConvBlock(128 + 96, 64)
        self.decode0 = VISTAXLConvBlock(64, 32)
        self.refine = VISTAXLConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def to_channels_first(feature: torch.Tensor) -> torch.Tensor:
        return feature.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def upsample_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        f1, f2, f3, f4 = self.encoder(x)

        f1 = self.to_channels_first(f1)
        f2 = self.to_channels_first(f2)
        f3 = self.to_channels_first(f3)
        f4 = self.to_channels_first(f4)

        batch_size, channels, height, width = f4.shape

        tokens = f4.flatten(2).transpose(1, 2)
        tokens = self.mlstm(tokens)

        f4 = tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )

        x = self.upsample_to(f4, f3)
        x = self.decode3(torch.cat([x, f3], dim=1))

        x = self.upsample_to(x, f2)
        x = self.decode2(torch.cat([x, f2], dim=1))

        x = self.upsample_to(x, f1)
        x = self.decode1(torch.cat([x, f1], dim=1))

        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = self.decode0(x)

        x = F.interpolate(
            x, size=input_size, mode="bilinear", align_corners=False
        )
        x = self.refine(x)

        return self.output(x)
