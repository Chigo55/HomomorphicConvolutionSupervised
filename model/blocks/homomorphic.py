import torch
import torch.nn as nn


class RGB2YCrCbBlock(nn.Module):
    def __init__(self, offset: float, trainable: bool) -> None:
        super().__init__()
        self.offset = nn.Parameter(
            data=torch.tensor(data=offset), requires_grad=trainable
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = (r - y) * 0.713 + self.offset
        cb = (b - y) * 0.564 + self.offset
        return y, cr, cb


class YCrCb2RGBBlock(nn.Module):
    def __init__(self, offset: float, trainable: bool) -> None:
        super().__init__()
        self.offset = nn.Parameter(
            data=torch.tensor(data=offset), requires_grad=trainable
        )
        self.r_cr = nn.Parameter(data=torch.tensor(data=1.403), requires_grad=trainable)
        self.g_cb = nn.Parameter(data=torch.tensor(data=-0.344), requires_grad=trainable)
        self.g_cr = nn.Parameter(data=torch.tensor(data=-0.714), requires_grad=trainable)
        self.b_cb = nn.Parameter(data=torch.tensor(data=1.773), requires_grad=trainable)

    def forward(
        self,
        lu: torch.Tensor,
        cr: torch.Tensor,
        cb: torch.Tensor,
    ) -> torch.Tensor:
        cr = cr - self.offset
        cb = cb - self.offset

        r = lu + self.r_cr * cr
        g = lu + self.g_cb * cb + self.g_cr * cr
        b = lu + self.b_cb * cb

        rgb = torch.cat(tensors=[r, g, b], dim=1)
        return rgb


class HomomorphicSeparationBlock(nn.Module):
    def __init__(self, raw_cutoff: float, trainable: bool) -> None:
        super().__init__()
        self.cutoff = self._cutoff_logit(raw_cutoff=raw_cutoff, trainable=trainable)

    def _cutoff_logit(self, raw_cutoff: float, trainable: bool) -> torch.Tensor:
        c = torch.tensor(data=raw_cutoff)
        c = torch.clamp(input=c, min=1e-5, max=0.5 - 1e-5)

        p = torch.clamp(input=2.0 * c, min=1e-5, max=1.0 - 1e-5)
        logit = torch.log(input=p / (1.0 - p))

        return nn.Parameter(data=logit, requires_grad=trainable)

    def _gaussian_lpf(
        self, size: tuple[int, int], cutoff: torch.Tensor
    ) -> torch.Tensor:
        height, width = size
        fy = torch.fft.fftfreq(height, d=1.0).to(device=cutoff.device)
        fx = torch.fft.fftfreq(width, d=1.0).to(device=cutoff.device)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        y, x = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(input=x * x + y * y).to(device=cutoff.device)

        cutoff = 0.5 * torch.sigmoid(input=cutoff)
        cutoff = torch.clamp(input=cutoff, min=1e-5, max=0.5 - 1e-5)

        sigma = cutoff / torch.sqrt(
            input=2.0 * torch.log(input=torch.tensor(data=2.0, device=cutoff.device))
        )
        h = torch.exp(input=-(radius ** 2) / (2.0 * sigma ** 2))
        h = h.unsqueeze(dim=0).unsqueeze(dim=0)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2:]

        x_log = torch.log(input=torch.clamp(input=x, min=1e-5))

        x_fft = torch.fft.fft2(x_log, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft)

        h = self._gaussian_lpf(size=(height, width), cutoff=self.cutoff)
        low_fft = x_fft * h

        low_log = torch.fft.ifft2(torch.fft.ifftshift(low_fft), norm="ortho").real
        high_log = x_log - low_log

        low = torch.exp(input=low_log)
        high = torch.exp(input=high_log)
        return low, high


class ImageDecomposition(nn.Module):
    def __init__(
        self,
        offset: float,
        raw_cutoff: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.rgb2ycrcb = RGB2YCrCbBlock(offset=offset, trainable=trainable)
        self.homomorphic = HomomorphicSeparationBlock(
            raw_cutoff=raw_cutoff, trainable=trainable
        )
        self.ycrcb2rgb = YCrCb2RGBBlock(offset=offset, trainable=trainable)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lu, cr, cb = self.rgb2ycrcb(x)
        il, re = self.homomorphic(lu)
        return lu, cr, cb, il, re


class ImageComposition(nn.Module):
    def __init__(self, offset: float, trainable: bool) -> None:
        super().__init__()
        self.ycrcb2rgb = YCrCb2RGBBlock(offset=offset, trainable=trainable)

    def forward(
        self,
        cr: torch.Tensor,
        cb: torch.Tensor,
        il: torch.Tensor,
        re: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lu = il * re
        rgb = self.ycrcb2rgb(lu, cr, cb)
        return rgb, lu
