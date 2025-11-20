from typing import Any, Final

from engine.engine import LightningEngine
from model.model import LowLightEnhancerLightning


def get_hparams() -> dict[str, Any]:
    hparams: dict[str, Any] = {
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size": 256,
        "batch_size": 1,
        "num_workers": 10,
        "seed": 42,
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "log_every_n_steps": 5,
        "log_dir": "runs/",
        "experiment_name": "valid/",
        "inference": "inference/",
        "patience": 20,
        "hidden_channels": 64,
        "num_resolution": 4,
        "dropout_ratio": 0.1,
        "offset": 0.5,
        "cutoff": 0.25,
        "trainable": False,
        "device": "cuda",
        "optim": "adam",
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
