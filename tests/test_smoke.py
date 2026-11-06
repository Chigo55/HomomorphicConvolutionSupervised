from pathlib import Path

import mlflow
import mlflow.artifacts
import pytest
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


def small_cfg(tmp_path: Path, *extra: str):
    """Tiny CPU config over the synthetic splits in tmp_path."""
    paths = {
        "data.train_dir": tmp_path / "train",
        "data.valid_dir": tmp_path / "valid",
        "data.bench_dir": tmp_path / "no_bench",
        "tracking.uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "tracking.artifact_dir": tmp_path / "mlartifacts",
    }
    quoted = [
        f"{k}='{v.as_posix() if isinstance(v, Path) else v}'" for k, v in paths.items()
    ]
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose(
            "config",
            overrides=quoted
            + [
                "device=cpu",
                "train.precision=fp32",
                "train.lr=1e-3",
                "train.log_every_n_steps=1",
                "tracking.system_metrics=false",
                "data.image_size=32",
                "data.batch_size=2",
                "data.num_workers=0",
                "model.embed_dim=8",
                "model.dropout_ratio=0.0",
                *extra,
            ],
        )


def fit_runs(cfg) -> list:
    client = mlflow.MlflowClient(cfg.tracking.uri)
    experiment = client.get_experiment_by_name(cfg.experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    return [r for r in runs if r.data.tags.get("mode") == "fit"]


def train_losses(cfg, run_id: str) -> list[float]:
    history = mlflow.MlflowClient(cfg.tracking.uri).get_metric_history(
        run_id, "train/loss/total"
    )
    return [m.value for m in sorted(history, key=lambda m: (m.timestamp, m.step))]


def test_fit_overfits_a_few_images_and_records_the_run(tmp_path):
    write_pairs(tmp_path / "train", count=4)
    write_pairs(tmp_path / "valid", count=2)
    cfg = small_cfg(tmp_path, "train.max_epochs=10")

    fit(cfg)

    [run] = fit_runs(cfg)
    run_id = run.info.run_id
    losses = train_losses(cfg, run_id)
    # measured: ~23% drop when learning, within ±2% noise at lr=0
    assert losses[-1] < 0.9 * losses[0]
    assert "valid/psnr" in run.data.metrics
    client = mlflow.MlflowClient(cfg.tracking.uri)
    saved = {a.path for a in client.list_artifacts(run_id, "checkpoints")}
    assert {"checkpoints/best.pt", "checkpoints/last.pt"} <= saved
    [log] = client.list_artifacts(run_id, "logs")
    assert "[EPOCH 10/10]" in mlflow.artifacts.load_text(f"runs:/{run_id}/{log.path}")


def test_resume_continues_the_same_run(tmp_path):
    write_pairs(tmp_path / "train", count=4)
    write_pairs(tmp_path / "valid", count=2)
    cfg = small_cfg(tmp_path, "train.max_epochs=2", "train.checkpoint_every=1")
    fit(cfg)
    [run] = fit_runs(cfg)
    run_id = run.info.run_id
    first = train_losses(cfg, run_id)  # 2 epochs x 2 steps
    checkpoints = f"runs:/{run_id}/checkpoints"

    # from the epoch-1 Checkpoint, epoch 2 replays exactly
    fit(small_cfg(tmp_path, f"resume='{checkpoints}/epoch-001.pt'"))
    replayed = train_losses(cfg, run_id)[len(first) :]
    assert replayed == pytest.approx(first[2:])

    # a finished Run has nothing left to train
    fit(small_cfg(tmp_path, f"resume='{checkpoints}/last.pt'"))
    assert len(train_losses(cfg, run_id)) == len(first) + 2

    # train.max_epochs may be raised to extend it
    fit(small_cfg(tmp_path, f"resume='{checkpoints}/last.pt'", "train.max_epochs=3"))
    assert len(train_losses(cfg, run_id)) == len(first) + 4
    assert [r.info.run_id for r in fit_runs(cfg)] == [run_id]
