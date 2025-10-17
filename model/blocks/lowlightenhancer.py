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
        trainable: bool,
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
            out_channels=1,
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

        cr_restored, cb_restored, re_restored = self.feature_restorer(
            cr,
            cb,
            re,
        )

        il_enh: Tensor = self.illumination_enhancer(il)

        img_enh, y_enh = self.composition(
            cr_restored,
            cb_restored,
            il_enh,
            re_restored,
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
                "luminance": y_enh,
                "chroma_red": cr_restored,
                "chroma_blue": cb_restored,
                "illuminance": il_enh,
                "reflectance": re_restored,
                "rgb": img_enh,
            },
        }
        return outputs
