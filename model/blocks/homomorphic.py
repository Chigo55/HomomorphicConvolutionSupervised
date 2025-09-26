from typing import Tuple, TypeAlias

import torch
import torch.nn as nn

YCrCbComponents: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
IlluminationPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
DecompositionOutputs: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
CompositionOutputs: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class RGB2YCrCbBlock(nn.Module):
    def __init__(self, offset: float, trainable: bool) -> None:
        super().__init__()
        self.offset: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=offset), requires_grad=trainable
        )

    def forward(self, x: torch.Tensor) -> YCrCbComponents:
        r: torch.Tensor = x[:, 0:1, :, :]
        g: torch.Tensor = x[:, 1:2, :, :]
        b: torch.Tensor = x[:, 2:3, :, :]

        lu: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
        cr: torch.Tensor = 0.5 * r - 0.418688 * g - 0.081312 * b + self.offset
        cb: torch.Tensor = -0.168736 * r - 0.331264 * g + 0.5 * b + self.offset
        return lu, cr, cb


class YCrCb2RGBBlock(nn.Module):
    def __init__(self, offset: float, trainable: bool) -> None:
        super().__init__()
        self.offset: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=offset), requires_grad=trainable
        )
        self.r_cr: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=1.5748), requires_grad=trainable
        )
        self.g_cb: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=-0.1873), requires_grad=trainable
        )
        self.g_cr: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=-0.4681), requires_grad=trainable
        )
        self.b_cb: nn.Parameter = nn.Parameter(
            data=torch.tensor(data=1.8556), requires_grad=trainable
        )

    def forward(
        self,
        lu: torch.Tensor,
        cr: torch.Tensor,
        cb: torch.Tensor,
    ) -> torch.Tensor:
        cr = cr - self.offset
        cb = cb - self.offset

        r: torch.Tensor = lu + self.r_cr * cr
        g: torch.Tensor = lu + self.g_cb * cb + self.g_cr * cr
        b: torch.Tensor = lu + self.b_cb * cb

        rgb: torch.Tensor = torch.cat(tensors=[r, g, b], dim=1)
        return rgb


class HomomorphicSeparationBlock(nn.Module):
    def __init__(self, raw_cutoff: float, eps: float, trainable: bool) -> None:
        super().__init__()
        self.eps: float = eps
        self.trainable: bool = trainable

        if trainable:
            self.cutoff: torch.Tensor = self._cutoff_logit(
                raw_cutoff=raw_cutoff, trainable=trainable
            )
        else:
            cutoff: torch.Tensor = self._cutoff_logit(
                raw_cutoff=raw_cutoff, trainable=trainable
            )
            self.register_buffer(name="cutoff", tensor=cutoff)
            self.cutoff = cutoff

    def _cutoff_logit(self, raw_cutoff: float, trainable: bool) -> torch.Tensor:
        c: torch.Tensor = torch.tensor(data=raw_cutoff)
        c = torch.clamp(input=c, min=self.eps, max=0.5 - self.eps)

        p: torch.Tensor = torch.clamp(input=2.0 * c, min=self.eps, max=1.0 - self.eps)
        logit: torch.Tensor = torch.log(input=p / (1.0 - p))

        if trainable:
            logit = nn.Parameter(data=logit, requires_grad=True)
            return logit
        return logit

    def _gaussian_lpf(
        self, size: Tuple[int, int], cutoff: torch.Tensor
    ) -> torch.Tensor:
        height, width = size
        fy: torch.Tensor = torch.fft.fftfreq(height, d=1.0).to(device=cutoff.device)
        fx: torch.Tensor = torch.fft.fftfreq(width, d=1.0).to(device=cutoff.device)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        y, x = torch.meshgrid(fy, fx, indexing="ij")
        radius: torch.Tensor = torch.sqrt(input=x * x + y * y).to(device=cutoff.device)

        cutoff = 0.5 * torch.sigmoid(input=cutoff)
        cutoff = torch.clamp(input=cutoff, min=self.eps, max=0.5 - self.eps)

        sigma: torch.Tensor = cutoff / torch.sqrt(
            input=torch.log(input=torch.tensor(data=2.0, device=cutoff.device))
        )
        h: torch.Tensor = torch.exp(input=-(radius * radius) / (2.0 * sigma * sigma))
        h = h.unsqueeze(dim=0).unsqueeze(dim=0)
        return h

    def forward(self, x: torch.Tensor) -> IlluminationPair:
        height, width = x.shape[-2:]

        x_log: torch.Tensor = torch.log(input=torch.clamp(input=x, min=self.eps))

        x_fft: torch.Tensor = torch.fft.fft2(x_log, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft)

        h: torch.Tensor = self._gaussian_lpf(size=(height, width), cutoff=self.cutoff)
        low_fft: torch.Tensor = x_fft * h

        low_log: torch.Tensor = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft), norm="ortho"
        ).real
        high_log: torch.Tensor = x_log - low_log

        low: torch.Tensor = torch.exp(input=low_log)
        high: torch.Tensor = torch.exp(input=high_log)
        return low, high


class ImageDecomposition(nn.Module):
    def __init__(
        self,
        offset: float,
        raw_cutoff: float,
        eps: float = 1e-6,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.rgb2ycrcb: RGB2YCrCbBlock = RGB2YCrCbBlock(
            offset=offset, trainable=trainable
        )
        self.homomorphic: HomomorphicSeparationBlock = HomomorphicSeparationBlock(
            raw_cutoff=raw_cutoff, eps=eps, trainable=trainable
        )
        self.ycrcb2rgb: YCrCb2RGBBlock = YCrCb2RGBBlock(
            offset=offset, trainable=trainable
        )

    def forward(self, x: torch.Tensor) -> DecompositionOutputs:
        lu, cr, cb = self.rgb2ycrcb(x)
        il, re = self.homomorphic(lu)
        return lu, cr, cb, il, re


class ImageComposition(nn.Module):
    def __init__(self, offset: float, trainable: bool) -> None:
        super().__init__()
        self.ycrcb2rgb: YCrCb2RGBBlock = YCrCb2RGBBlock(
            offset=offset, trainable=trainable
        )

    def forward(
        self,
        cr: torch.Tensor,
        cb: torch.Tensor,
        il: torch.Tensor,
        re: torch.Tensor,
    ) -> CompositionOutputs:
        lu: torch.Tensor = il * re
        rgb: torch.Tensor = self.ycrcb2rgb(lu, cr, cb)
        return rgb, lu