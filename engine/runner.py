from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from lightning import LightningDataModule, LightningModule, Trainer
from torch import Tensor

from data.dataloader import LowLightDataModule
from utils.utils import save_images


class _BaseRunner(ABC):
    def __init__(
        self,
        model: LightningModule,
        trainer: Trainer,
        hparams: dict[str, Any],
    ) -> None:
        self.trainer: Trainer = trainer
        self.hparams: dict[str, Any] = hparams
        self.model: LightningModule = model

        self.datamodule: LightningDataModule = self._build_datamodule()

        self.log_dir = self.hparams.get("log_dir", "runs/")
        self.experiment_name = self.hparams.get("experiment_name", "test/")
        self.version = self.hparams.get("version", "version/")
        self.inference = self.hparams.get("inference", "inference/")
        self.out_dir: Path = (
            Path(self.log_dir) / self.experiment_name / self.version / self.inference
        )

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
        output = self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
        )
        if output is None:
            print("[WARN] Prediction returned None. Skipping save.")
            return

        else:
            output = cast(list[list[Tensor]], output)
            save_images(batch_list=output, out_dir=self.out_dir)
            print("[INFO] Inference Completed.")
