from typing import Any, TypeAlias

import lightning as L
import torch
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.asgd import ASGD
from torch.optim.lbfgs import LBFGS
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop
from torch.optim.rprop import Rprop
from torch.optim.sgd import SGD

from model.blocks.lowlightenhancer import LowLightEnhancer, EnhancerOutputs
from model.loss import MeanAbsoluteError, MeanSquaredError, StructuralSimilarity
from utils.metrics import ImageQualityMetrics, MetricDict

LossDict: TypeAlias = dict[str, torch.Tensor]
ResultDict: TypeAlias = EnhancerOutputs
Batch: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class LowLightEnhancerLightning(L.LightningModule):
    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model: LowLightEnhancer = LowLightEnhancer(
            hidden_channels=self.hparams.get("hidden_channels", 64),
            num_resolution=self.hparams.get("num_resolution", 4),
            dropout_ratio=self.hparams.get("dropout_ratio", 0.2),
            offset=self.hparams.get("offset", 0.5),
            raw_cutoff=self.hparams.get("raw_cutoff", 0.1),
            trainable=self.hparams.get("trainable", False),
        )

        self.mae_loss: MeanAbsoluteError = MeanAbsoluteError().eval()
        self.mse_loss: MeanSquaredError = MeanSquaredError().eval()
        self.ssim_loss: StructuralSimilarity = StructuralSimilarity().eval()

        self.metric: ImageQualityMetrics = ImageQualityMetrics().eval()

    def forward(self, low: torch.Tensor) -> ResultDict:
        return self.model(low)

    def _calculate_loss(self, results: ResultDict, target: torch.Tensor) -> LossDict:
        pred: torch.Tensor = results["enhanced"]["rgb"]

        loss_mae: torch.Tensor = self.mae_loss(pred, target)
        loss_mse: torch.Tensor = self.mse_loss(pred, target)
        loss_ssim: torch.Tensor = self.ssim_loss(pred, target)
        loss_total: torch.Tensor = loss_mae + loss_mse + loss_ssim

        losses: LossDict = {
            "mae": loss_mae,
            "mse": loss_mse,
            "ssim": loss_ssim,
            "total": loss_total,
        }
        return losses

    def _shared_step(self, batch: Batch) -> tuple[LossDict, ResultDict]:
        low_img, high_img = batch
        results: ResultDict = self.forward(low=low_img)
        losses: LossDict = self._calculate_loss(results=results, target=high_img)
        return losses, results

    def _log_media(self, stage: str, results: ResultDict, batch_idx: int) -> None:
        if batch_idx % 50 != 0:
            return

        low = results["low"]
        enhanced = results["enhanced"]

        for i, (key, val) in enumerate(iterable=low.items()):
            self.logger.experiment.add_images(
                f"{stage}/low/{i + 1}_{key}", val, self.global_step
            )
        for i, (key, val) in enumerate(iterable=enhanced.items()):
            self.logger.experiment.add_images(
                f"{stage}/enhanced/{i + 1}_{key}", val, self.global_step
            )

    def _log_losses(self, stage: str, losses: LossDict) -> None:
        log_dict: dict[str, torch.Tensor] = {}
        for i, (key, val) in enumerate(iterable=losses.items()):
            log_dict[f"{stage}/{i + 1}_{key}"] = val

        self.log_dict(dictionary=log_dict, prog_bar=True)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        losses, results = self._shared_step(batch=batch)

        self._log_media(stage="train", results=results, batch_idx=batch_idx)
        self._log_losses(stage="train", losses=losses)

        return losses["total"]

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        losses, results = self._shared_step(batch=batch)

        self._log_media(stage="valid", results=results, batch_idx=batch_idx)
        self._log_losses(stage="valid", losses=losses)

        return losses["total"]

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        low_img, high_img = batch

        results: ResultDict = self.forward(low=low_img)
        metrics: MetricDict = self.metric.full(
            preds=results["enhanced"]["rgb"], targets=high_img
        )

        self.log_dict(
            dictionary={
                "test/1_PSNR": metrics["PSNR"],
                "test/2_SSIM": metrics["SSIM"],
                "test/3_LPIPS": metrics["LPIPS"],
                "test/4_NIQE": metrics["NIQE"],
                "test/5_BRISQUE": metrics["BRISQUE"],
            },
            prog_bar=True,
        )

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        low_img, _ = batch

        results: ResultDict = self.forward(low=low_img)
        return results["enhanced"]["rgb"]

    def configure_optimizers(self) -> Optimizer:
        optim_name: str = self.hparams["optim"].lower()

        if optim_name == "sgd":
            return SGD(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1e-2),
                momentum=self.hparams.get("momentum", 0.9),
                weight_decay=self.hparams.get("weight_decay", 1e-4),
                nesterov=self.hparams.get("nesterov", True),
            )

        if optim_name == "asgd":
            return ASGD(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1e-2),
                lambd=self.hparams.get("lambd", 1e-4),
                alpha=self.hparams.get("alpha", 0.75),
                t0=self.hparams.get("t0", 1e6),
                weight_decay=self.hparams.get("weight_decay", 1e-4),
            )

        if optim_name == "rmsprop":
            return RMSprop(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                alpha=self.hparams.get("alpha", 0.99),
                eps=self.hparams.get("eps", 1e-8),
                momentum=self.hparams.get("momentum", 0.9),
                weight_decay=self.hparams.get("weight_decay", 0),
            )

        if optim_name == "rprop":
            return Rprop(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1e-2),
                etas=self.hparams.get("etas", (0.5, 1.2)),
                step_sizes=self.hparams.get("step_sizes", (1e-6, 50)),
            )

        if optim_name == "adam":
            return Adam(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                betas=self.hparams.get("betas", (0.9, 0.999)),
                eps=self.hparams.get("eps", 1e-8),
                weight_decay=self.hparams.get("weight_decay", 0),
            )

        if optim_name == "adamw":
            return AdamW(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                betas=self.hparams.get("betas", (0.9, 0.999)),
                eps=self.hparams.get("eps", 1e-8),
                weight_decay=self.hparams.get("weight_decay", 1e-2),
            )

        if optim_name == "adamax":
            return Adamax(
                params=self.parameters(),
                lr=self.hparams.get("lr", 2e-3),
                betas=self.hparams.get("betas", (0.9, 0.999)),
                eps=self.hparams.get("eps", 1e-8),
                weight_decay=self.hparams.get("weight_decay", 0),
            )

        if optim_name == "adagrad":
            return Adagrad(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1e-2),
                lr_decay=self.hparams.get("lr_decay", 0),
                weight_decay=self.hparams.get("weight_decay", 0),
                eps=self.hparams.get("eps", 1e-10),
            )

        if optim_name == "adadelta":
            return Adadelta(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1.0),
                rho=self.hparams.get("rho", 0.9),
                eps=self.hparams.get("eps", 1e-6),
                weight_decay=self.hparams.get("weight_decay", 0),
            )

        if optim_name == "lbfgs":
            return LBFGS(
                params=self.parameters(),
                lr=self.hparams.get("lr", 1.0),
                max_iter=self.hparams.get("max_iter", 20),
                max_eval=self.hparams.get("max_eval"),
                tolerance_grad=self.hparams.get("tolerance_grad", 1e-7),
                tolerance_change=self.hparams.get("tolerance_change", 1e-9),
                history_size=self.hparams.get("history_size", 100),
                line_search_fn=self.hparams.get("line_search_fn"),
            )

        raise ValueError(f"Unsupported optimizer: {optim_name}")