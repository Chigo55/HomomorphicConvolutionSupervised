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
        hidden_channels: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.conv1: ResidualBlock = ResidualBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            dropout_ratio=dropout_ratio,
        )
        self.conv2: ResidualBlock = ResidualBlock(
            in_channels=hidden_channels,
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


class FeatureRestorationBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.conv_cr: DoubleConv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            dropout_ratio=dropout_ratio,
        )
        self.conv_cb: DoubleConv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            dropout_ratio=dropout_ratio,
        )
        self.conv_re: DoubleConv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            dropout_ratio=dropout_ratio,
        )

    def forward(
        self,
        cr: torch.Tensor,
        cb: torch.Tensor,
        re: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cr = self.conv_cr(cr)
        cb = self.conv_cb(cb)
        re = self.conv_re(re)
        return cr, cb, re
