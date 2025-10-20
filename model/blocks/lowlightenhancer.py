import torch
import torch.nn as nn
from torch import Tensor

from model.blocks.featurerestorer import FeatureRestorationBlock
from model.blocks.homomorphic import ImageComposition, ImageDecomposition
from model.blocks.illuminationenhancer import IlluminationEnhancer


class LowLightEnhancer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_resolution: int,
        dropout_ratio: float,
        cutoff: float,
        offset: float,
    ) -> None:
        super().__init__()

        self.decomposition: ImageDecomposition = ImageDecomposition(
            offset=offset,
            cutoff=cutoff,
        )
        self.feature_restorer: FeatureRestorationBlock = FeatureRestorationBlock(
            in_channels=1,
            out_channels=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_ratio=dropout_ratio,
        )

        self.illumination_enhancer: IlluminationEnhancer = IlluminationEnhancer(
            in_channels=1,
            out_channels=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_resolution=num_resolution,
            dropout_ratio=dropout_ratio,
        )

        self.composition: ImageComposition = ImageComposition(
            offset=offset,
        )

    def forward(self, low: Tensor) -> dict[str, dict[str, Tensor]]:
        y, cr, cb, il, re = self.decomposition(low)

        cr_restored, cb_restored = self.feature_restorer(
            cr,
            cb,
        )

        y_enhenced: Tensor = self.illumination_enhancer(il, re)

        img_enh = self.composition(
            cr_restored,
            cb_restored,
            y_enhenced,
        )
        img_enh = torch.clamp(input=img_enh, min=0.0, max=1.0)

        outputs: dict[str, dict[str, Tensor]] = {
            "low": {
                "luminance": y,
                "chroma_red": cr,
                "chroma_blue": cb,
                "illuminance": il,
                "reflectance": re,
                "rgb": low,
            },
            "enhanced": {
                "luminance": y_enhenced,
                "chroma_red": cr_restored,
                "chroma_blue": cb_restored,
                "rgb": img_enh,
            },
        }
        return outputs
