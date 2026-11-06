import math
import os
import random
import tempfile
from pathlib import Path

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset

from data.dataset import build_datasets, make_loader, manifest
from engine import tracking
from engine.evaluate import autocast, quality_metrics, run_bench, validate
from model.blocks import LowLightEnhancer
from model.loss import EnhancementLoss


class NonFiniteLossError(RuntimeError):
    pass


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer, warmup: int, total: int
) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def fit(cfg: DictConfig) -> None:
    """Train one Run, keep best/last/periodic Checkpoints as MLflow artifacts,
    then benchmark the best Checkpoint."""
    ckpt = None
    if cfg.resume:
        ckpt = torch.load(
            tracking.resolve_checkpoint(cfg.resume, cfg.tracking.uri),
            map_location="cpu",
        )
        cfg = OmegaConf.create(ckpt["config"])  # a Run has exactly one configuration
    t = cfg.train
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    train_sets = build_datasets(
        cfg.data.train_dir, crop_size=cfg.data.image_size, augment=True
    )
    valid_sets = build_datasets(cfg.data.valid_dir)
    if not train_sets or not valid_sets:
        raise FileNotFoundError(
            f"Need datasets in {cfg.data.train_dir} and {cfg.data.valid_dir}"
        )
    train_loader = make_loader(
        ConcatDataset(train_sets), cfg.data.batch_size, True, cfg.data.num_workers
    )
    valid_loader = make_loader(
        ConcatDataset(valid_sets), 1, False, cfg.data.num_workers
    )

    model = LowLightEnhancer(**cfg.model).to(device)
    loss_fn = EnhancementLoss(device=cfg.device)
    metrics = quality_metrics(("psnr", "ssim"), cfg.device)
    optimizer = Adam(
        model.parameters(),
        lr=t.lr,
        betas=tuple(t.betas),
        eps=t.eps,
        weight_decay=t.weight_decay,
    )
    total_steps = t.max_epochs * len(train_loader)
    scheduler = cosine_with_warmup(
        optimizer, max(1, int(total_steps * t.warmup_ratio)), total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=t.precision == "fp16")
    state = {"epoch": 0, "step": 0, "best": None, "bad_epochs": 0}
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
    improved = (lambda a, b: a > b) if t.monitor_mode == "max" else (lambda a, b: a < b)

    run_id = ckpt["run_id"] if ckpt else None
    with (
        tempfile.TemporaryDirectory() as staging,
        tracking.start_run(cfg, run_id=run_id, tags={"mode": "fit"}) as run,
    ):
        run_id = run.info.run_id
        if ckpt is None:
            tracking.log_config(cfg)
            mlflow.log_params(
                {
                    "data.train_manifest": manifest(cfg.data.train_dir),
                    "data.valid_manifest": manifest(cfg.data.valid_dir),
                }
            )
            tracking.log_code_state()

        def save(name: str) -> None:
            path = Path(staging) / f"{name}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "state": state,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "run_id": run_id,
                },
                path,
            )
            mlflow.log_artifact(str(path), "checkpoints")

        try:
            for epoch in range(state["epoch"], t.max_epochs):
                model.train()
                for batch in train_loader:
                    low = batch["low"].to(device, non_blocking=True)
                    normal = batch["normal"].to(device, non_blocking=True)
                    with autocast(device, t.precision):
                        losses = loss_fn(model(low)["enhanced"]["rgb"], normal)
                    loss = losses["total"]
                    if not torch.isfinite(loss):
                        raise NonFiniteLossError(
                            f"loss is {loss.item()} at step {state['step']}"
                        )

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    log_now = state["step"] % t.log_every_n_steps == 0
                    if log_now:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float("inf")
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if log_now:
                        logs = {f"train/loss/{k}": v.item() for k, v in losses.items()}
                        logs["train/lr"] = scheduler.get_last_lr()[0]
                        logs["train/grad_norm"] = grad_norm.item()
                        mlflow.log_metrics(logs, step=state["step"])
                    state["step"] += 1

                scores, samples = validate(
                    model,
                    valid_loader,
                    loss_fn,
                    metrics,
                    device,
                    t.precision,
                    t.num_sample_images,
                )
                mlflow.log_metrics(scores, step=state["step"])
                for i, image in enumerate(samples):
                    mlflow.log_image(image, key=f"valid/sample_{i}", step=epoch)
                if t.monitor not in scores:
                    raise KeyError(
                        f"train.monitor={t.monitor!r} is not one of {sorted(scores)}"
                    )

                score = scores[t.monitor]
                state["epoch"] = epoch + 1
                is_best = state["best"] is None or improved(score, state["best"])
                if is_best:
                    state["best"], state["bad_epochs"] = score, 0
                    save("best")
                else:
                    state["bad_epochs"] += 1
                save("last")
                if (epoch + 1) % t.checkpoint_every == 0:
                    save(f"epoch-{epoch + 1:03d}")
                print(
                    f"[EPOCH {epoch + 1}/{t.max_epochs}] {t.monitor}={score:.4f} best={state['best']:.4f}"
                )
                if state["bad_epochs"] >= t.patience:
                    print(
                        f"[INFO] Early stopping: no {t.monitor} improvement in {t.patience} epochs."
                    )
                    break
        except Exception as e:
            tracking.notify(
                f"[crashed] {cfg.experiment_name} run {run_id}: {type(e).__name__}: {e}"
            )
            raise
        tracking.notify(
            f"[finished] {cfg.experiment_name} run {run_id}: best {t.monitor}={state['best']:.4f} "
            f"after {state['epoch']} epochs"
        )
    run_bench(cfg, f"runs:/{run_id}/checkpoints/best.pt")  # after the fit Run has ended
