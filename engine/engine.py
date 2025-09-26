from typing import List, Optional, Type

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
        model: Type[LightningModule],
        hparams: dict,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.hparams = hparams
        self.checkpoint_path = checkpoint_path

        seed_everything(seed=self.hparams["seed"], workers=True)

        self.logger = self._build_logger()
        self.callbacks = self._build_callbacks()

        self.trainer = self._set_build_trainer()

    def _set_build_trainer(self) -> Trainer:
        return Trainer(
            max_epochs=self.hparams["max_epochs"],
            accelerator=self.hparams["accelerator"],
            devices=self.hparams["devices"],
            precision=self.hparams["precision"],
            log_every_n_steps=self.hparams["log_every_n_steps"],
            logger=self.logger,
            callbacks=self.callbacks,
        )

    def _build_logger(self) -> TensorBoardLogger:
        return TensorBoardLogger(
            save_dir=self.hparams["log_dir"], name=self.hparams["experiment_name"]
        )

    def _build_callbacks(self) -> List[Callback]:
        return [
            ModelCheckpoint(
                monitor="valid/5_tot",
                save_top_k=1,
                mode="min",
                filename="best-{epoch:02d}",
            ),
            ModelCheckpoint(
                every_n_epochs=1,
                save_top_k=-1,
                filename="epoch-{epoch:02d}",
            ),
            EarlyStopping(
                monitor="valid/5_tot",
                patience=self.hparams["patience"],
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
            checkpoint_path=self.checkpoint_path,
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