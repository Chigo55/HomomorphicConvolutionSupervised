import pyiqa
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EnhancementLoss(nn.Module):
    """MAE + MSE + (1 - colour SSIM) between Enhanced and Normal-light images."""

    def __init__(self, device: str) -> None:
        super().__init__()
        self.ssim = pyiqa.create_metric("ssimc", device=device, as_loss=True)

    def forward(self, enhanced: Tensor, normal: Tensor) -> dict[str, Tensor]:
        mae = F.l1_loss(enhanced, normal)
        mse = F.mse_loss(enhanced, normal)
        ssim = 1 - self.ssim(enhanced, normal).mean()
        return {"mae": mae, "mse": mse, "ssim": ssim, "total": mae + mse + ssim}
