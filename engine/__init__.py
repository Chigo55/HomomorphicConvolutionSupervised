from typing import Optional, Type

from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from engine.runner import (
    LightningBenchmarker,
    LightningInferencer,
    LightningTrainer,
    LightningValidater,
    _BaseRunner,
)


class LightningEngine:
    def __init__(
        self,
        model_class: Type[LightningModule],
        hparams: dict,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.hparams = hparams
        self.checkpoint_path = checkpoint_path

        seed_everything(seed=self.hparams.get("seed", 42), workers=True)

        if checkpoint_path:
            self.model = model_class.load_from_checkpoint(checkpoint_path)
        else:
            self.model = model_class(hparams=self.hparams)

        self.logger = self._build_logger()
        self.callbacks = self._build_callbacks()
        self.trainer = self._build_trainer()

    def _build_trainer(self) -> Trainer:
        return Trainer(
            max_epochs=self.hparams.get("max_epochs", 100),
            accelerator=self.hparams.get("accelerator", "gpu"),
            devices=self.hparams.get("devices", 1),
            precision=self.hparams.get("precision", "32-true"),
            log_every_n_steps=self.hparams.get("log_every_n_steps", 5),
            logger=self.logger,
            callbacks=self.callbacks,
            benchmark=False,
            deterministic=True
        )

    def _build_logger(self) -> TensorBoardLogger:
        return TensorBoardLogger(
            save_dir=self.hparams.get("log_dir", "runs/"),
            name=self.hparams.get("experiment_name", "test/"),
        )

    def _build_callbacks(self) -> list[Callback]:
        return [
            ModelCheckpoint(
                monitor="valid/4_total",
                save_top_k=1,
                mode="min",
                filename="best-{epoch:02d}",
            ),
            ModelCheckpoint(
                every_n_epochs=20,
                save_top_k=-1,
                filename="epoch-{epoch:02d}",
            ),
            EarlyStopping(
                monitor="valid/4_total",
                patience=self.hparams.get("patience", 20),
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

    def _create_and_run_runner(self, runner_class: Type[_BaseRunner]) -> None:
        runner = runner_class(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
        )
        runner.run()

    def train(self) -> None:
        self._create_and_run_runner(runner_class=LightningTrainer)

    def valid(self) -> None:
        self._create_and_run_runner(runner_class=LightningValidater)

    def bench(self) -> None:
        self._create_and_run_runner(runner_class=LightningBenchmarker)

    def infer(self) -> None:
        self._create_and_run_runner(runner_class=LightningInferencer)
