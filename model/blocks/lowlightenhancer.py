from typing import TypeAlias

import torch
import torch.nn as nn

from model.blocks.featurerestorer import FeatureRestorationBlock
from model.blocks.homomorphic import (
    DecompositionOutputs,
    ImageComposition,
    ImageDecomposition,
)
from model.blocks.illuminationenhancer import IlluminationEnhancer

TensorDict: TypeAlias = dict[str, torch.Tensor]
EnhancerOutputs: TypeAlias = dict[str, TensorDict]


class LowLightEnhancer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_resolution: int,
        dropout_ratio: float,
        raw_cutoff: float,
        offset: float,
        trainable: bool,
    ) -> None:
        super().__init__()

        self.decomposition: ImageDecomposition = ImageDecomposition(
            offset=offset,
            raw_cutoff=raw_cutoff,
            trainable=trainable,
        )
        self.feature_restorer: FeatureRestorationBlock = FeatureRestorationBlock(
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden_channels * num_resolution,
            dropout_ratio=dropout_ratio,
        )

        self.illumination_enhancer: IlluminationEnhancer = IlluminationEnhancer(
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden_channels,
            num_resolution=num_resolution,
            dropout_ratio=dropout_ratio,
            trainable=trainable,
        )

        self.composition: ImageComposition = ImageComposition(
            offset=offset, trainable=trainable
        )

    def forward(self, low: torch.Tensor) -> EnhancerOutputs:
        decomposition_outputs: DecompositionOutputs = self.decomposition(low)
        luminance, chroma_red, chroma_blue, illuminance, reflectance = (
            decomposition_outputs
        )
        enhanced_chroma_red, enhanced_chroma_blue, enhanced_reflectance = (
            self.feature_restorer(chroma_red, chroma_blue, reflectance)
        )
        enhanced_illuminance: torch.Tensor = self.illumination_enhancer(illuminance)
        enhanced_img, enhanced_luminance = self.composition(
            enhanced_chroma_red,
            enhanced_chroma_blue,
            enhanced_illuminance,
            enhanced_reflectance,
        )
        enhanced_img = torch.clamp(input=enhanced_img, min=0.0, max=1.0)

        results: EnhancerOutputs = {
            "low": {
                "luminance": luminance,
                "chroma_red": chroma_red,
                "chroma_blue": chroma_blue,
                "illuminance": illuminance,
                "reflectance": reflectance,
                "rgb": low,
            },
            "enhanced": {
                "luminance": enhanced_luminance,
                "chroma_red": enhanced_chroma_red,
                "chroma_blue": enhanced_chroma_blue,
                "illuminance": enhanced_illuminance,
                "reflectance": enhanced_reflectance,
                "rgb": enhanced_img,
            },
        }
        return results
