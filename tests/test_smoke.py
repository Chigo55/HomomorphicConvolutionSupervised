import torch

from model.blocks import LowLightEnhancer


def tiny_enhancer() -> LowLightEnhancer:
    return LowLightEnhancer(
        embed_dim=8,
        num_heads=2,
        mlp_ratio=1,
        num_resolution=2,
        dropout_ratio=0.0,
        offset=0.5,
        cutoff=0.1,
    )


def test_enhanced_image_matches_low_light_shape_and_unit_range():
    torch.manual_seed(0)
    low_light = torch.rand(2, 3, 16, 16)

    enhanced = tiny_enhancer()(low_light)["enhanced"]["rgb"]

    assert enhanced.shape == low_light.shape
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0


def test_every_parameter_contributes_to_the_enhanced_image():
    torch.manual_seed(0)
    model = tiny_enhancer()

    model(torch.rand(2, 3, 16, 16))["enhanced"]["rgb"].mean().backward()

    dead = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert dead == []
