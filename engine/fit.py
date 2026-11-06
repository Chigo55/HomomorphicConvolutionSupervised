import math
import operator
import os
import random
import tempfile
from pathlib import Path

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms.functional import to_pil_image

from data.dataset import build_datasets, make_loader, manifest
from engine import tracking
from engine.evaluate import PRECISIONS, bench, enhance, quality_metrics
from model.blocks import LowLightEnhancer
from model.loss import EnhancementLoss

# What a resumed Run may change: where it runs and for how long. Everything else
# stays as the Run was first configured.
RESUME_OVERRIDES = ("device", "tracking", "data.num_workers", "train.max_epochs")


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(rng: dict) -> None:
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer, warmup: int, total: int
) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)


def _resume_config(cfg: DictConfig, saved: dict) -> tuple[DictConfig, dict]:
    """The saved config, with RESUME_OVERRIDES keys the command line set
    (values that differ from conf/config.yaml) taken from `cfg`. Also returns
    the overrides that change the Run."""
    defaults = tracking.flatten(OmegaConf.load(tracking.REPO / "conf" / "config.yaml"))
    requested = {
        k: v
        for k, v in tracking.flatten(cfg).items()
        if k not in ("mode", "ckpt", "resume") and defaults.get(k) != v
    }
    original = tracking.flatten(OmegaConf.create(saved))

    def overridable(key: str) -> bool:
        return any(key == o or key.startswith(f"{o}.") for o in RESUME_OVERRIDES)

    resumed = OmegaConf.create(saved)
    for k, v in requested.items():
        if overridable(k):
            OmegaConf.update(resumed, k, v, merge=False)
    resumed.resume = cfg.resume
    ignored = sorted(
        k for k, v in requested.items() if not overridable(k) and original.get(k) != v
    )
    if ignored:
        print(
            f"[WARN] Resume keeps the Run's original {ignored}; "
            f"only {list(RESUME_OVERRIDES)} can change."
        )
    changed = {
        k: v for k, v in requested.items() if overridable(k) and original.get(k) != v
    }
    return resumed, changed


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics: dict[str, nn.Module],
    precision: str,
    num_samples: int,
) -> tuple[dict[str, float], list[Image.Image]]:
    """Loss terms and metrics averaged over the whole Validation set, plus the
    first `num_samples` images as low-light | enhanced | normal-light strips."""
    model.eval()
    device = next(model.parameters()).device
    sums: dict[str, float] = {}
    samples = []
    for batch in loader:
        low_light = batch["low_light"].to(device)
        normal_light = batch["normal_light"].to(device)
        enhanced = enhance(model, low_light, precision)
        losses = loss_fn(enhanced, normal_light)
        scores = {f"valid/loss/{k}": v.item() for k, v in losses.items()}
        for k, m in metrics.items():
            scores[f"valid/{k}"] = m(enhanced, normal_light).mean().item()
        for k, v in scores.items():
            sums[k] = sums.get(k, 0.0) + v * len(low_light)
        if len(samples) < num_samples:
            strip = torch.cat([low_light[0], enhanced[0], normal_light[0]], dim=-1)
            samples.append(to_pil_image(strip.cpu()))
    count = len(loader.dataset)
    return {k: v / count for k, v in sums.items()}, samples


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    train_cfg: DictConfig,
    state: dict,
) -> None:
    model.train()
    device = next(model.parameters()).device
    for batch in loader:
        low_light = batch["low_light"].to(device, non_blocking=True)
        normal_light = batch["normal_light"].to(device, non_blocking=True)
        losses = loss_fn(enhance(model, low_light, train_cfg.precision), normal_light)
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is {loss.item()} at step {state['step']}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        log_now = state["step"] % train_cfg.log_every_n_steps == 0
        if log_now:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if log_now:
            logs = {f"train/loss/{k}": v.item() for k, v in losses.items()}
            logs["train/lr"] = scheduler.get_last_lr()[0]
            logs["train/grad_norm"] = grad_norm.item()
            mlflow.log_metrics(logs, step=state["step"])
        state["step"] += 1


def fit(cfg: DictConfig) -> None:
    """Train one Run, keep best/last/periodic Checkpoints as MLflow artifacts,
    then benchmark the best Checkpoint."""
    ckpt, changed = None, {}
    if cfg.resume:
        ckpt = tracking.fetch_checkpoint(cfg.resume, cfg.tracking.uri)
        cfg, changed = _resume_config(cfg, ckpt["config"])
    train_cfg = cfg.train
    if ckpt and (
        ckpt["state"]["epoch"] >= train_cfg.max_epochs
        or ckpt["state"]["bad_epochs"] >= train_cfg.patience
    ):
        print("[INFO] This Run already finished; use mode=bench to benchmark again.")
        return
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
        lr=train_cfg.lr,
        betas=tuple(train_cfg.betas),
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
    )
    # ponytail: raising max_epochs on resume reshapes the remaining cosine curve
    total_steps = train_cfg.max_epochs * len(train_loader)
    scheduler = cosine_with_warmup(
        optimizer, max(1, int(total_steps * train_cfg.warmup_ratio)), total_steps
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=PRECISIONS[train_cfg.precision] is torch.float16
    )
    state = {"epoch": 0, "step": 0, "best": None, "bad_epochs": 0}
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
    improved = operator.gt if train_cfg.monitor_mode == "max" else operator.lt

    run_id = ckpt["run_id"] if ckpt else None
    try:
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
            else:  # params are immutable, so resume overrides become tags
                mlflow.set_tags({f"resume.{k}": str(v) for k, v in changed.items()})
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
                        "rng": _rng_state(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "run_id": run_id,
                    },
                    path,
                )
                mlflow.log_artifact(str(path), "checkpoints")

            if ckpt:  # last, so model init above can't consume the restored stream
                _set_rng_state(ckpt["rng"])
            for epoch in range(state["epoch"], train_cfg.max_epochs):
                _train_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    scheduler,
                    scaler,
                    train_cfg,
                    state,
                )
                scores, samples = validate(
                    model,
                    valid_loader,
                    loss_fn,
                    metrics,
                    train_cfg.precision,
                    train_cfg.num_sample_images,
                )
                mlflow.log_metrics(scores, step=state["step"])
                for i, image in enumerate(samples):
                    mlflow.log_image(image, key=f"valid/sample_{i}", step=state["step"])
                if train_cfg.monitor not in scores:
                    raise KeyError(
                        f"train.monitor={train_cfg.monitor!r} is not one of {sorted(scores)}"
                    )

                score = scores[train_cfg.monitor]
                state["epoch"] = epoch + 1
                if state["best"] is None or improved(score, state["best"]):
                    state["best"], state["bad_epochs"] = score, 0
                    save("best")
                else:
                    state["bad_epochs"] += 1
                save("last")
                if (epoch + 1) % train_cfg.checkpoint_every == 0:
                    save(f"epoch-{epoch + 1:03d}")
                print(
                    f"[EPOCH {epoch + 1}/{train_cfg.max_epochs}] "
                    f"{train_cfg.monitor}={score:.4f} best={state['best']:.4f}"
                )
                if state["bad_epochs"] >= train_cfg.patience:
                    print(
                        f"[INFO] Early stopping: no {train_cfg.monitor} improvement "
                        f"in {train_cfg.patience} epochs."
                    )
                    break
        bench(cfg, f"runs:/{run_id}/checkpoints/best.pt")  # after the fit Run ended
    except Exception as e:
        tracking.notify(
            f"[crashed] {cfg.experiment_name} run {run_id}: {type(e).__name__}: {e}"
        )
        raise
    tracking.notify(
        f"[finished] {cfg.experiment_name} run {run_id}: "
        f"best {train_cfg.monitor}={state['best']:.4f} after {state['epoch']} epochs"
    )
