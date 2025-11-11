import random
from typing import Any

from engine.engine import LightningEngine
from model.model import LowLightEnhancerLightning


def get_hparams() -> dict[str, Any]:
    hparams: dict[str, Any] = {
        # Engine
        "seed": 42,
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "16-mixed",
        "log_every_n_steps": 5,
        "log_dir": "runs/",
        "experiment_name": "add_vit/",
        "patience": 100,
        # Runner
        "inference": "inference/",
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size": 512,
        "batch_size": 4,
        "num_workers": 10,
        # Model
        "embed_dim": 64,
        "num_heads": 8,
        "mlp_ratio": 4,
        "num_resolution": 4,
        "dropout_ratio": 0.2,
        "offset": 0.5,
        "cutoff": 0.1,
    }
    return hparams


def main() -> None:
    hparams: dict[str, Any] = get_hparams()
    seed: int = random.randint(0, 1000)

    hparams["seed"] = seed
    engine: LightningEngine = LightningEngine(
        model_class=LowLightEnhancerLightning,
        hparams=hparams,
    )
    engine.train()
    engine.valid()
    engine.bench()


if __name__ == "__main__":
    main()
