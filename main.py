import os
import random
from typing import Any, Dict, List

from engine import LightningEngine
from model.model import LowLightEnhancerLightning

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

HParams = Dict[str, Any]


def get_hparams() -> HParams:
    hparams: HParams = {
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size": 256,
        "batch_size": 12,
        "num_workers": 10,
        "seed": 42,
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "32-true",
        "log_every_n_steps": 5,
        "log_dir": "runs/",
        "experiment_name": "test/",
        "inference": "inference/",
        "patience": 20,
        "hidden_channels": 64,
        "num_resolution": 4,
        "dropout_ratio": 0.2,
        "offset": 0.5,
        "raw_cutoff": 0.25,
        "trainable": False,
        "device": "cuda",
        "optim": "sgd",
    }
    return hparams


def main() -> None:
    hparams: HParams = get_hparams()
    opts: List[str] = [
        "sgd",
        "asgd",
        "rmsprop",
        "rprop",
        "adam",
        "adamw",
        "adamax",
        "adadelta",
    ]
    seed: int = random.randint(0, 1000)

    for opt in opts:
        print(f"\n[STARTING] Optimizer: {opt}")
        hparams["experiment_name"] = opt
        hparams["trainable"] = False
        hparams["seed"] = seed

        engine = LightningEngine(
            model_class=LowLightEnhancerLightning,
            hparams=hparams,
        )

        engine.train()
        engine.valid()
        engine.bench()

    for opt in opts:
        print(f"\n[STARTING] Optimizer: {opt}")
        hparams["experiment_name"] = opt
        hparams["trainable"] = True
        hparams["seed"] = seed

        engine = LightningEngine(
            model_class=LowLightEnhancerLightning,
            hparams=hparams,
        )

        engine.train()
        engine.valid()
        engine.bench()


if __name__ == "__main__":
    main()
