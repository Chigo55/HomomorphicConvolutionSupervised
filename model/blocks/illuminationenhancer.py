import torch
import torch.nn as nn
from torch import Tensor

from model.blocks.attention import SelfAttentionBlock


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
        x1: Tensor = self.conv1(self.dropout1(self.act1(self.bn1(x))))
        x2: Tensor = self.conv2(self.dropout2(self.act2(self.bn2(x1))))

        residual: Tensor = self.skip_proj(x) + x2
        return residual


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
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

    def flatten(self, x):
        return x.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)

    def unflatten(self, x, h, w):
        return x.permute(0, 2, 1).unflatten(dim=2, sizes=(h, w))

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.flatten(x=x)
        x = self.attn(x)
        x = self.unflatten(x=x, h=h, w=w)
        x = self.conv2(x)
        return x


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )


    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.conv(x)


class IlluminationEnhancer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_resolution: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()

        self.in_conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        dim_level = embed_dim
        down: list[nn.Module] = []
        for level in range(num_resolution):
            down.append(
                DoubleConv(
                    in_channels=dim_level,
                    embed_dim=dim_level,
                    out_channels=dim_level,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_ratio=dropout_ratio,
                )
            )
            down.append(
                Downsampling(
                    in_channels=dim_level,
                    out_channels=dim_level * 2,
                )
            )
            dim_level *= 2
        self.down: nn.ModuleList = nn.ModuleList(modules=down)

        mid: list[nn.Module] = []
        for _ in range(num_resolution // 2):
            mid.append(
                DoubleConv(
                    in_channels=dim_level,
                    embed_dim=dim_level,
                    out_channels=dim_level,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_ratio=dropout_ratio,
                )
            )
        self.mid: nn.ModuleList = nn.ModuleList(modules=mid)

        up: list[nn.Module] = []
        for level in range(num_resolution):
            up.append(
                Upsampling(
                    in_channels=dim_level,
                    out_channels=dim_level // 2,
                )
            )
            up.append(
                DoubleConv(
                    in_channels=dim_level,
                    embed_dim=dim_level,
                    out_channels=dim_level // 2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_ratio=dropout_ratio,
                )
            )
            dim_level //= 2

        self.up: nn.ModuleList = nn.ModuleList(modules=up)

        self.out_conv: nn.Conv2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.in_conv(x)

        residuals: list[Tensor] = []
        for module in self.down:
            if isinstance(module, Downsampling):
                residuals.append(x)
                x = module(x)
            else:
                x = module(x)

        for module in self.mid:
            x = module(x)

        for module in self.up:
            if isinstance(module, Upsampling):
                x = module(x)
                x = torch.cat(tensors=[x, residuals.pop()], dim=1)
            else:
                x = module(x)

        return self.out_conv(x)
