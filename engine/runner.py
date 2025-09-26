from abc import ABC, abstractmethod
from typing import Any, TypeAlias, cast

from lightning import LightningModule, Trainer

from data.dataloader import LowLightDataModule
from utils.utils import TensorBatches, save_images

HParamsDict: TypeAlias = dict[str, Any]
PredictResults: TypeAlias = TensorBatches


class _BaseRunner(ABC):
    def __init__(
        self,
        model: LightningModule,
        trainer: Trainer,
        hparams: HParamsDict,
    ) -> None:
        self.trainer: Trainer = trainer
        self.hparams: HParamsDict = hparams
        self.log_dir: str = self.hparams.get("log_dir", "runs/")
        self.experiment_name: str = self.hparams.get("experiment_name", "test/")
        self.inference: str = self.hparams.get("inference", "inference/")
        self.save_dir: str = self.log_dir + self.experiment_name + self.inference

        self.model: LightningModule = model
        self.datamodule: LowLightDataModule = self._build_datamodule()

    def _build_datamodule(self) -> LowLightDataModule:
        datamodule: LowLightDataModule = LowLightDataModule(
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
            datamodule=self.datamodule,
        )
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
        results: PredictResults = cast(
            PredictResults,
            self.trainer.predict(
                model=self.model,
                datamodule=self.datamodule,
            ),
        )
        save_images(results=results, save_dir=self.save_dir)
        print("[INFO] Inference Completed.")
