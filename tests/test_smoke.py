from pathlib import Path

import mlflow
import torch
from hydra import compose, initialize_config_dir
from torchvision.transforms.functional import to_pil_image

from engine.fit import fit
from model.blocks import LowLightEnhancer

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


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

    dead = [
        n for n, p in model.named_parameters() if p.grad is None or not p.grad.any()
    ]
    assert dead == []


def write_pairs(split: Path, count: int, size: int = 32) -> None:
    """Normal-light images and Low-light copies at 20% brightness."""
    for sub in ("low", "high"):
        (split / "synth" / sub).mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    for i in range(count):
        normal = torch.rand(3, size, size, generator=g)
        to_pil_image(normal).save(split / "synth" / "high" / f"{i}.png")
        to_pil_image(normal * 0.2).save(split / "synth" / "low" / f"{i}.png")


def test_fit_overfits_a_few_images_and_records_the_run(tmp_path):
    write_pairs(tmp_path / "train", count=4)
    write_pairs(tmp_path / "valid", count=2)
    overrides = {
        "data.train_dir": tmp_path / "train",
        "data.valid_dir": tmp_path / "valid",
        "data.bench_dir": tmp_path / "no_bench",
        "tracking.uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "tracking.artifact_dir": tmp_path / "mlartifacts",
    }
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                f"{k}='{Path(v).as_posix() if isinstance(v, Path) else v}'"
                for k, v in overrides.items()
            ]
            + [
                "device=cpu",
                "train.precision=fp32",
                "train.max_epochs=10",
                "train.lr=1e-3",
                "train.log_every_n_steps=1",
                "tracking.system_metrics=false",
                "data.image_size=32",
                "data.batch_size=2",
                "data.num_workers=0",
                "model.embed_dim=8",
                "model.dropout_ratio=0.0",
            ],
        )

    fit(cfg)

    client = mlflow.MlflowClient(cfg.tracking.uri)
    experiment = client.get_experiment_by_name(cfg.experiment_name)
    [run] = client.search_runs([experiment.experiment_id])
    run_id = run.info.run_id
    losses = [m.value for m in client.get_metric_history(run_id, "train/loss/total")]
    # measured: ~23% drop when learning, within ±2% noise at lr=0
    assert losses[-1] < 0.9 * losses[0]
    assert "valid/psnr" in run.data.metrics
    saved = {a.path for a in client.list_artifacts(run_id, "checkpoints")}
    assert {"checkpoints/best.pt", "checkpoints/last.pt"} <= saved
