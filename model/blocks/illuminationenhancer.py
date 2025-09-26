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
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.act1 = nn.SiLU()
        self.dropout1= nn.Dropout(p=dropout_ratio)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = nn.SiLU()
        self.dropout2= nn.Dropout(p=dropout_ratio)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(self.dropout1(self.act1(self.bn1(x))))
        x2 = self.conv2(self.dropout2(self.act2(self.bn2(x1))))

        residual = self.skip_proj(x) + x2
        return residual


class DoubleConv(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        dropout_ratio: float
    ) -> None:
        super().__init__()

        self.conv1 = ResidualBlock(in_channels=in_channels, out_channels=out_channels, dropout_ratio=dropout_ratio)
        self.conv2 = ResidualBlock(in_channels=out_channels, out_channels=out_channels, dropout_ratio=dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Downsampling(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        trainable: bool
    ) -> None:
        super().__init__()
        self.trainable = trainable

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            return self.down(self.conv(x))
        else:
            return self.down(x)


class Upsampling(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        trainable: bool
    ) -> None:
        super().__init__()
        self.trainable = trainable

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            return self.up(self.conv(x))
        else:
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

        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        in_ch, out_ch = 0, 0
        down = []
        for d in range(num_resolution):
            in_ch = hidden_channels * (2**d)
            out_ch = hidden_channels * (2 ** (d + 1))
            down.append(
                DoubleConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    dropout_ratio=dropout_ratio
                )
            )
            down.append(Downsampling(in_channels=out_ch, out_channels=out_ch, trainable=trainable))
        self.down = nn.ModuleList(modules=down)

        mid = []
        for i in range(num_resolution//2):
            m_ch = hidden_channels * (2 ** num_resolution)
            mid.append(
                DoubleConv(
                    in_channels=m_ch,
                    out_channels=m_ch,
                    dropout_ratio=dropout_ratio
                )
            )
        self.mid = nn.ModuleList(modules=mid)

        up = []

        for u in reversed(range(num_resolution)):
            in_ch = hidden_channels * (2 ** (u + 1))
            out_ch = hidden_channels * (2**u)
            up.append(Upsampling(in_channels=in_ch, out_channels=in_ch, trainable=trainable))
            up.append(
                DoubleConv(
                    in_channels=in_ch * 2,
                    out_channels=out_ch,
                    dropout_ratio=dropout_ratio
                )
            )
        self.up = nn.ModuleList(modules=up)

        self.out_conv = nn.Conv2d(
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

        r = []
        for m in self.down:
            if isinstance(m, Downsampling):
                r.append(x)
                x = m(x)
            else:
                x = m(x)

        for m in self.mid:
            x = m(x)

        for m in self.up:
            if isinstance(m, Upsampling):
                x = m(x)
                x = torch.cat(tensors=[x, r.pop()], dim=1)
            else:
                x = m(x)

        return self.out_conv(x)
