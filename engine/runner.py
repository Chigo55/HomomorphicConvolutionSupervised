from abc import ABC, abstractmethod
from typing import Optional, Type

from lightning import LightningModule, Trainer

from data.dataloader import LowLightDataModule
from utils.utils import save_images


class _BaseRunner(ABC):
    def __init__(
        self,
        model: Type[LightningModule],
        trainer: Trainer,
        hparams: dict,
        checkpoint_path: Optional[str] = None,
        default_checkpoint_name: Optional[str] = None,
    ) -> None:
        self.trainer = trainer
        self.hparams = hparams
        self.save_dir = hparams["log_dir"] + hparams["experiment_name"]

        if checkpoint_path:
            self.model = model.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location="cpu",
            )
            self.checkpoint_path = checkpoint_path
        else:
            self.model = model(hparams=hparams)
            self.checkpoint_path = default_checkpoint_name

        self.datamodule = self._build_datamodule()

    def _build_datamodule(self) -> LowLightDataModule:
        datamodule = LowLightDataModule(
            train_dir=self.hparams["train_data_path"],
            valid_dir=self.hparams["valid_data_path"],
            bench_dir=self.hparams["bench_data_path"],
            infer_dir=self.hparams["infer_data_path"],
            image_size=self.hparams["image_size"],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
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
            ckpt_path=self.checkpoint_path,
        )
        print("[INFO] Training Completed.")


class LightningValidater(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Validating...")
        self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.checkpoint_path,
        )
        print("[INFO] Validation Completed.")


class LightningBenchmarker(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Benchmarking...")
        self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.checkpoint_path,
        )
        print("[INFO] Benchmark Completed.")


class LightningInferencer(_BaseRunner):
    def run(self) -> None:
        print("[INFO] Start Inferencing...")
        results = self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.checkpoint_path,
        )
        save_images(
            results=results,
            save_dir=self.save_dir + self.hparams["inference"],
        )
        print("[INFO] Inference Completed.")