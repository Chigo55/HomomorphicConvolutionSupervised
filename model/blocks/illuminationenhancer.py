import torch
import torch.nn as nn


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
        x: torch.Tensor,
    ) -> torch.Tensor:
        x1: torch.Tensor = self.conv1(self.dropout1(self.act1(self.bn1(x))))
        x2: torch.Tensor = self.conv2(self.dropout2(self.act2(self.bn2(x1))))

        residual: torch.Tensor = self.skip_proj(x) + x2
        return residual


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.conv1: ResidualBlock = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_ratio=dropout_ratio,
        )
        self.conv2: ResidualBlock = ResidualBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            dropout_ratio=dropout_ratio,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Downsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.trainable: bool = trainable

        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down: nn.AvgPool2d = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.trainable:
            return self.down(self.conv(x))
        return self.down(x)


class Upsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.trainable: bool = trainable

        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.up: nn.Upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.trainable:
            return self.up(self.conv(x))
        return self.up(x)


class IlluminationEnhancer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_resolution: int,
        dropout_ratio: float,
        trainable: bool,
    ) -> None:
        super().__init__()

        self.in_conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        in_ch: int = 0
        out_ch: int = 0
        down: list[nn.Module] = []
        for depth in range(num_resolution):
            in_ch = hidden_channels * (2**depth)
            out_ch = hidden_channels * (2 ** (depth + 1))
            down.append(
                DoubleConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    dropout_ratio=dropout_ratio,
                )
            )
            down.append(
                Downsampling(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    trainable=trainable,
                )
            )
        self.down: nn.ModuleList = nn.ModuleList(modules=down)

        mid: list[nn.Module] = []
        for _ in range(num_resolution // 2):
            mid_channels: int = hidden_channels * (2**num_resolution)
            mid.append(
                DoubleConv(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    dropout_ratio=dropout_ratio,
                )
            )
        self.mid: nn.ModuleList = nn.ModuleList(modules=mid)

        up: list[nn.Module] = []
        for level in reversed(range(num_resolution)):
            in_ch = hidden_channels * (2 ** (level + 1))
            out_ch = hidden_channels * (2**level)
            up.append(
                Upsampling(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    trainable=trainable,
                )
            )
            up.append(
                DoubleConv(
                    in_channels=in_ch * 2,
                    out_channels=out_ch,
                    dropout_ratio=dropout_ratio,
                )
            )
        self.up: nn.ModuleList = nn.ModuleList(modules=up)

        self.out_conv: nn.Conv2d = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_conv(x)

        residuals: list[torch.Tensor] = []
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
