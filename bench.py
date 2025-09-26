from engine import LightningEngine
from model.model import LowLightEnhancerLightning

import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"


def get_hparams():
    hparams = {
        # 데이터 모듈
        "train_data_path" : "data/1_train",
        "valid_data_path" : "data/2_valid",
        "bench_data_path" : "data/3_bench",
        "infer_data_path" : "data/4_infer",
        "image_size" : 256,
        "batch_size" : 12,
        "num_workers" : 10,
        # 엔진
        "seed" : 42,
        "max_epochs" : 100,
        "accelerator" : "gpu",
        "devices" : 1,
        "precision" : "32-true",
        "log_every_n_steps" : 5,
        "log_dir" : "runs/",
        "experiment_name" : "test/",
        "inference" : "inference/",
        "patience" : 25,
        # 모델 구조
        "hidden_channels" : 64,
        "num_resolution" : 4,
        "dropout_ratio" : 0.2,
        "offset" : 0.5,
        "raw_cutoff" : 0.25,
        "trainable" : False,
        # 모델 모듈
        "device" : "cuda",
        "optim" : "sgd",
    }
    return hparams


def main():
    hparams = get_hparams()
    opts = ["sgd", "asgd", "rmsprop", "rprop", "adam", "adamw", "adamax", "adadelta"]
    path = Path("./runs/").glob("**/version_0/checkpoints/best-epoch=*.ckpt")

    for p in path:
        engin = LightningEngine(model_class=LowLightEnhancerLightning, hparams=hparams, checkpoint_path=p)

        # engin.train()
        engin.valid()
        engin.bench()
        engin.infer()


if __name__ == "__main__":
    main()