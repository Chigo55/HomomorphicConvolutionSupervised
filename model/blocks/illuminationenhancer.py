import torch
import torch.nn as nn
from torch import Tensor

from model.blocks.attention import CrossAttentionBlock
from model.utils import Flatten, Unflatten


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
        self.down: nn.ModuleList = nn.ModuleList(modules=[])
        for level in range(num_resolution):
            self.down.append(
                module=nn.ModuleList(
                    modules=[
                        CrossAttentionBlock(
                            embed_dim=dim_level,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout_ratio=dropout_ratio,
                        ),
                        Downsampling(
                            in_channels=dim_level,
                            out_channels=dim_level * 2,
                        ),
                        Downsampling(
                            in_channels=dim_level,
                            out_channels=dim_level * 2,
                        ),
                    ]
                )
            )
            dim_level *= 2

        self.mid: nn.ModuleList = nn.ModuleList(modules=[])
        for _ in range(num_resolution // 2):
            self.mid.append(
                module=CrossAttentionBlock(
                    embed_dim=dim_level,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_ratio=dropout_ratio,
                )
            )

        self.up: nn.ModuleList = nn.ModuleList(modules=[])
        for level in range(num_resolution):
            self.up.append(
                module=nn.ModuleList(
                    modules=[
                        Upsampling(
                            in_channels=dim_level,
                            out_channels=dim_level // 2,
                        ),
                        nn.Conv2d(
                            in_channels=dim_level,
                            out_channels=dim_level // 2,
                            kernel_size=1,
                            stride=1,
                            bias=False,
                        ),
                        CrossAttentionBlock(
                            embed_dim=dim_level // 2,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout_ratio=dropout_ratio,
                        ),
                    ]
                )
            )
            dim_level //= 2

        self.out_conv: nn.Conv2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x_res = x
        c_res = c
        x = self.in_conv(x)
        c = self.in_conv(c)

        x_lst = []
        c_lst = []
        for cttn, x_down, c_down in self.down:
            _, _, h, w = x.shape
            x, c = Flatten(x=x), Flatten(x=c)
            x = cttn(x, c)
            x, c = Unflatten(x=x, h=h, w=w), Unflatten(x=c, h=h, w=w)

            x_lst.append(x)
            c_lst.append(c)

            x = x_down(x)
            c = c_down(c)

        for cttn in self.mid:
            _, _, h, w = x.shape
            x, c = Flatten(x=x), Flatten(x=c)
            x = cttn(x, c)
            x, c = Unflatten(x=x, h=h, w=w), Unflatten(x=c, h=h, w=w)

        for x_up, fusion, cttn in self.up:
            x = x_up(x)
            x = fusion(torch.cat(tensors=[x, x_lst.pop()], dim=1))
            c = c_lst.pop()

            _, _, h, w = x.shape
            x, c = Flatten(x=x), Flatten(x=c)
            x = cttn(x, c)
            x, c = Unflatten(x=x, h=h, w=w), Unflatten(x=c, h=h, w=w)

        out = self.out_conv(x) + x_res * c_res

        return out
