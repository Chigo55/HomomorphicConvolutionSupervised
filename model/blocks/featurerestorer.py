from typing import Tuple

import torch.nn as nn
from torch import Tensor

from model.blocks.attention import SelfAttentionBlock
from model.blocks.flatten import Flatten, Unflatten


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_features=in_channels)
        self.act1: nn.SiLU = nn.SiLU()
        self.dropout1: nn.Dropout = nn.Dropout(p=dropout_ratio)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_features=out_channels)
        self.act2: nn.SiLU = nn.SiLU()
        self.dropout2: nn.Dropout = nn.Dropout(p=dropout_ratio)
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if in_channels != out_channels:
            self.skip_proj: nn.Module = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.skip_proj = nn.Identity()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        res = x
        x = self.conv1(self.dropout1(self.act1(self.bn1(x))))
        x = self.conv2(self.dropout2(self.act2(self.bn2(x))))
        x = self.skip_proj(res) + x

        return x


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.conv1: ResidualBlock = ResidualBlock(
            in_channels=in_channels,
            out_channels=embed_dim,
            dropout_ratio=dropout_ratio,
        )
        self.attn = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )
        self.conv2: ResidualBlock = ResidualBlock(
            in_channels=embed_dim,
            out_channels=out_channels,
            dropout_ratio=dropout_ratio,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = Flatten(x=x)
        x = self.attn(x)
        x = Unflatten(x=x, h=h, w=w)
        x = self.conv2(x)
        return x


class FeatureRestorationBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.cr_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )
        self.cb_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

    def forward(
        self,
        cr: Tensor,
        cb: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        cr = self.cr_conv(cr)
        cb = self.cr_conv(cb)
        return cr, cb
