from typing import Any, Final

from engine.engine import LightningEngine
from model.model import LowLightEnhancerLightning


def get_hparams() -> dict[str, Any]:
    hparams: dict[str, Any] = {
        # Engine
        "seed": 42,
        "max_epochs": 100,
        "accelerator": "auto",
        "devices": 1,
        "precision": "16-mixed",
        "log_every_n_steps": 5,
        "monitor_metric": "valid/3_total",
        "monitor_mode": "min",
        "patience": 20,
        # Runner
        "log_dir": "runs/",
        "experiment_name": "train/",
        "inference": "inference/",
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size": 256,
        "batch_size": 4,
        "num_workers": 10,
        # Model
        "embed_dim": 16,
        "num_heads": 2,
        "mlp_ratio": 1,
        "num_resolution": 2,
        "dropout_ratio": 0.2,
        "offset": 0.5,
        "cutoff": 0.1,
        "log_period": 25,
    }
    return hparams


DEFAULT_CHECKPOINT: Final[str] = (
    "./runs/test/lightning_logs/version_0/checkpoints/best-epoch=81.ckpt"
)


def main() -> None:
    hparams: dict[str, Any] = get_hparams()

    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        hparams=hparams,
        checkpoint_path=DEFAULT_CHECKPOINT,
    )

    engine.valid()


if __name__ == "__main__":
    main()
