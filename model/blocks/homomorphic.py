from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RGB2YCrCbBlock(nn.Module):
    def __init__(
        self,
        offset: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.offset: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=offset),
            requires_grad=trainable,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r: Tensor = x[:, 0:1, :, :]
        g: Tensor = x[:, 1:2, :, :]
        b: Tensor = x[:, 2:3, :, :]

        y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
        cr: Tensor = 0.5 * r - 0.418688 * g - 0.081312 * b
        cb: Tensor = -0.168736 * r - 0.331264 * g + 0.5 * b

        cr = cr + self.offset
        cb = cb + self.offset
        return y, cr, cb


class YCrCb2RGBBlock(nn.Module):
    def __init__(
        self,
        offset: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.offset: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=offset),
            requires_grad=trainable,
        )

    def forward(
        self,
        y: Tensor,
        cr: Tensor,
        cb: Tensor,
    ) -> Tensor:
        cr = cr - self.offset
        cb = cb - self.offset

        r: Tensor = y + (1.403 * cr)
        g: Tensor = y + (-0.344 * cb) + (-0.714 * cr)
        b: Tensor = y + (1.773 * cb)

        rgb: Tensor = torch.cat(tensors=[r, g, b], dim=1)
        return rgb


class HomomorphicSeparationBlock(nn.Module):
    def __init__(
        self,
        raw_cutoff: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.cutoff: Tensor = self._cutoff_logit(
            raw_cutoff=raw_cutoff,
            trainable=trainable,
        )

    def _cutoff_logit(
        self,
        raw_cutoff: float,
        trainable: bool,
    ) -> Tensor:
        c: Tensor = torch.tensor(data=raw_cutoff)
        c = torch.clamp(input=c, min=1e-5, max=0.5 - 1e-5)

        p: Tensor = torch.clamp(input=2.0 * c, min=1e-5, max=1.0 - 1e-5)
        logit: Tensor = torch.log(input=p / (1.0 - p))

        return nn.Parameter(
            data=logit,
            requires_grad=trainable,
        )

    def _gaussian_lpf(
        self,
        size: tuple[int, int],
        cutoff: Tensor,
    ) -> Tensor:
        height, width = size
        fy: Tensor = torch.fft.fftfreq(height, d=1.0).to(device=cutoff.device)
        fx: Tensor = torch.fft.fftfreq(width, d=1.0).to(device=cutoff.device)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        y, x = torch.meshgrid(fy, fx, indexing="ij")
        radius: Tensor = torch.sqrt(input=x * x + y * y).to(device=cutoff.device)

        cutoff = 0.5 * torch.sigmoid(input=cutoff)
        cutoff = torch.clamp(input=cutoff, min=1e-5, max=0.5 - 1e-5)

        sigma: Tensor = cutoff / torch.sqrt(
            input=torch.log(input=torch.tensor(data=2.0, device=cutoff.device))
        )
        h: Tensor = torch.exp(input=-(radius * radius) / (2.0 * sigma * sigma))
        h = h.unsqueeze(dim=0).unsqueeze(dim=0)
        return h

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        height, width = x.shape[-2:]

        x_log: Tensor = torch.log(input=torch.clamp(input=x, min=1e-5))

        x_fft: Tensor = torch.fft.fft2(x_log, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft)

        h: Tensor = self._gaussian_lpf(size=(height, width), cutoff=self.cutoff)
        low_fft: Tensor = x_fft * h

        low_log: torch.Tensor = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft), norm="ortho"
        ).real
        high_log: torch.Tensor = x_log - low_log

        il: Tensor = torch.exp(input=low_log)  # illuminance
        re: Tensor = torch.exp(input=high_log)  # reflectance
        return il, re


class ImageDecomposition(nn.Module):
    def __init__(
        self,
        offset: float,
        raw_cutoff: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.rgb2ycrcb: RGB2YCrCbBlock = RGB2YCrCbBlock(
            offset=offset,
            trainable=trainable,
        )
        self.homomorphic: HomomorphicSeparationBlock = HomomorphicSeparationBlock(
            raw_cutoff=raw_cutoff,
            trainable=trainable,
        )
        self.ycrcb2rgb: YCrCb2RGBBlock = YCrCb2RGBBlock(
            offset=offset,
            trainable=trainable,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        y, cr, cb = self.rgb2ycrcb(x)
        il, re = self.homomorphic(y)
        return y, cr, cb, il, re


class ImageComposition(nn.Module):
    def __init__(
        self,
        offset: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.ycrcb2rgb: YCrCb2RGBBlock = YCrCb2RGBBlock(
            offset=offset,
            trainable=trainable,
        )

    def forward(
        self,
        cr: Tensor,
        cb: Tensor,
        il: Tensor,
        re: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        y_enh: Tensor = il * re
        img_enh: Tensor = self.ycrcb2rgb(y_enh, cr, cb)
        return img_enh, y_enh
