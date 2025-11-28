from typing import Any, Literal, cast

import lightning as L
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from transformers import get_cosine_schedule_with_warmup

from data.utils import LowLightSample
from model.blocks.lowlightenhancer import LowLightEnhancer
from model.loss import MeanAbsoluteError, MeanSquaredError, StructuralSimilarity
from utils.metrics import ImageQualityMetrics


class LowLightEnhancerLightning(L.LightningModule):
    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model: LowLightEnhancer = LowLightEnhancer(
            embed_dim=self.hparams.get("embed_dim", 32),
            num_heads=self.hparams.get("num_heads", 8),
            mlp_ratio=self.hparams.get("mlp_ratio", 4),
            num_resolution=self.hparams.get("num_resolution", 4),
            dropout_ratio=self.hparams.get("dropout_ratio", 0.2),
            offset=self.hparams.get("offset", 0.5),
            cutoff=self.hparams.get("cutoff", self.hparams.get("cutoff", 0.1)),
        )

        self.mae_loss: MeanAbsoluteError = MeanAbsoluteError().eval()
        self.mse_loss: MeanSquaredError = MeanSquaredError().eval()
        self.ssim_loss: StructuralSimilarity = StructuralSimilarity().eval()

        self.metric: ImageQualityMetrics = ImageQualityMetrics().eval()

    def forward(
        self,
        low: Tensor,
    ) -> dict[str, dict[str, Tensor]]:
        return self.model(low)

    def _calculate_loss(
        self,
        outputs: dict[str, dict[str, Tensor]],
        target: Tensor,
    ) -> dict[str, Tensor]:
        pred_img: Tensor = outputs["enhanced"]["rgb"]

        loss_mae: Tensor = self.mae_loss(pred_img, target)
        loss_mse: Tensor = self.mse_loss(pred_img, target) * 0
        loss_ssim: Tensor = self.ssim_loss(pred_img, target)
        loss_total: Tensor = loss_mae + loss_mse + loss_ssim

        loss_dict: dict[str, Tensor] = {
            "mae": loss_mae,
            "mse": loss_mse,
            "ssim": loss_ssim,
            "total": loss_total,
        }
        return loss_dict

    def _shared_step(
        self,
        batch: LowLightSample,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor]]:
        low_img, high_img = batch
        outputs = self.forward(low=low_img)
        loss_dict = self._calculate_loss(outputs=outputs, target=high_img)
        return outputs, loss_dict

    def _logging(
        self,
        stage: Literal["train", "valid"],
        outputs: dict[str, dict[str, Tensor]],
        loss_dict: dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx % 50 != 0:
            return

        low = outputs["low"]
        enhanced = outputs["enhanced"]

        for i, (key, val) in enumerate(iterable=enhanced.items()):
            self.logger.experiment.add_images(
                f"{stage}/enhanced/{i + 1}_{key}", val, self.global_step
            )
        for i, (key, val) in enumerate(iterable=low.items()):
            self.logger.experiment.add_images(
                f"{stage}/low/{i + 1}_{key}", val, self.global_step
            )

        logs: dict[str, Tensor] = {}
        for i, (key, val) in enumerate(iterable=loss_dict.items()):
            logs[f"{stage}/{i + 1}_{key}"] = val

        self.log_dict(dictionary=logs, prog_bar=True)

    def training_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
    ) -> Tensor:
        outputs, loss_dict = self._shared_step(batch=batch)

        self._logging(
            stage="train",
            outputs=outputs,
            loss_dict=loss_dict,
            batch_idx=batch_idx,
        )

        return loss_dict["total"]

    def validation_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
    ) -> Tensor:
        outputs, loss_dict = self._shared_step(batch=batch)

        self._logging(
            stage="valid",
            outputs=outputs,
            loss_dict=loss_dict,
            batch_idx=batch_idx,
        )

        return loss_dict["total"]

    def test_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        low_img, high_img = batch

        outputs = self.forward(low=low_img)
        metrics = self.metric.full(preds=outputs["enhanced"]["rgb"], targets=high_img)

        self.log_dict(
            dictionary={
                "test/01_PSNR": metrics["PSNR"],
                "test/02_SSIM": metrics["SSIM"],
                "test/03_LPIPS": metrics["LPIPS"],
                "test/04_NIQE": metrics["NIQE"],
                "test/05_BRISQUE": metrics["BRISQUE"],
            },
            prog_bar=True,
        )

    def predict_step(
        self,
        batch: LowLightSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        low_img, _ = batch

        results = self.forward(low=low_img)
        return results["enhanced"]["rgb"]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        lr = float(self.hparams.get("lr", 1e-4))

        optimizer: Optimizer = Adam(
            params=self.parameters(),
            lr=lr,
            betas=self.hparams.get("betas", (0.9, 0.999)),
            eps=self.hparams.get("eps", 1e-8),
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )

        total_training_steps = cast(int, self.trainer.estimated_stepping_batches)

        warmup_ratio = 0.05
        num_warmup_steps = max(1, int(total_training_steps * warmup_ratio))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )

        sched_cfg: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [sched_cfg]
