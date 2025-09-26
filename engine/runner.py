from abc import ABC, abstractmethod

from lightning import LightningModule, Trainer

from data.dataloader import LowLightDataModule
from utils.utils import save_images


class _BaseRunner(ABC):
    def __init__(
        self,
        model: LightningModule,
        trainer: Trainer,
        hparams: dict,
    ) -> None:
        self.trainer = trainer
        self.hparams = hparams
        self.log_dir = self.hparams.get("log_dir", "runs/")
        self.experiment_name = self.hparams.get("experiment_name", "test/")
        self.inference = self.hparams.get("inference", "inference/")
        self.save_dir = self.log_dir +  self.experiment_name + self.inference

        self.model = model
        self.datamodule = self._build_datamodule()

    def _build_datamodule(self) -> LowLightDataModule:
        datamodule = LowLightDataModule(
            train_dir=self.hparams.get("train_data_path", "data/1_train"),
            valid_dir=self.hparams.get("valid_data_path", "data/2_valid"),
            bench_dir=self.hparams.get("bench_data_path", "data/3_bench"),
            infer_dir=self.hparams.get("infer_data_path", "data/4_infer"),
            image_size=self.hparams.get("image_size", 256),
            batch_size=self.hparams.get("batch_size", 16),
            num_workers=self.hparams.get("num_workers", 10),
        )

        return datamodule

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class LightningTrainer(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Training...")
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,        )
        print("[INFO] Training Completed.")


class LightningValidater(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Validating...")
        self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
        )
        print("[INFO] Validation Completed.")


class LightningBenchmarker(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Benchmarking...")
        self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
        )
        print("[INFO] Benchmark Completed.")


class LightningInferencer(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Inferencing...")
        results = self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
        )
        save_images(
            results=results,
            save_dir=self.save_dir
        )
        print("[INFO] Inference Completed.")
