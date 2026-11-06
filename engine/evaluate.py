import tempfile
import time
from pathlib import Path

import mlflow
import pyiqa
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from data.dataset import LowLightDataset, build_datasets, make_loader, manifest
from engine import tracking
from model.blocks import LowLightEnhancer

FULL_REFERENCE = ("psnr", "ssim", "lpips")
NO_REFERENCE = ("niqe", "brisque")


def autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[precision]
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)


def quality_metrics(names: tuple[str, ...], device: str) -> dict[str, nn.Module]:
    return {name: pyiqa.create_metric(name, device=device) for name in names}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics: dict[str, nn.Module],
    device: torch.device,
    precision: str,
    num_samples: int,
) -> tuple[dict[str, float], list[Image.Image]]:
    """Loss terms and metrics averaged over the whole Validation set, plus the
    first `num_samples` images as low-light | enhanced | normal-light strips."""
    model.eval()
    sums: dict[str, float] = {}
    samples = []
    for batch in loader:
        low = batch["low"].to(device)
        normal = batch["normal"].to(device)
        with autocast(device, precision):
            enhanced = model(low)["enhanced"]["rgb"]
            losses = loss_fn(enhanced, normal)
        enhanced = enhanced.float()
        scores = {f"valid/loss/{k}": v.item() for k, v in losses.items()}
        scores |= {
            f"valid/{k}": m(enhanced, normal).mean().item() for k, m in metrics.items()
        }
        for k, v in scores.items():
            sums[k] = sums.get(k, 0.0) + v * len(low)
        if len(samples) < num_samples:
            strip = torch.cat([low[0], enhanced[0], normal[0]], dim=-1)
            samples.append(to_pil_image(strip.cpu()))
    count = len(loader.dataset)
    return {k: v / count for k, v in sums.items()}, samples


def load_checkpoint(cfg: DictConfig, uri: str) -> tuple[nn.Module, dict, str]:
    """Model built from the checkpoint's own config, so architecture overrides
    on the command line can't mismatch the weights."""
    path = tracking.resolve_checkpoint(uri, cfg.tracking.uri)
    ckpt = torch.load(path, map_location="cpu")
    model = LowLightEnhancer(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(cfg.device).eval(), ckpt, path.stem


def bench(cfg: DictConfig) -> None:
    if not cfg.ckpt:
        raise ValueError("mode=bench needs ckpt=runs:/<run_id>/checkpoints/<name>.pt")
    run_bench(cfg, cfg.ckpt)


def run_bench(cfg: DictConfig, uri: str) -> None:
    """Benchmark results for one Checkpoint, as a child Run of its fit Run."""
    datasets = build_datasets(cfg.data.bench_dir)
    if not datasets:
        print(f"[WARN] No Benchmark set in {cfg.data.bench_dir}; skipping bench.")
        return
    model, ckpt, name = load_checkpoint(cfg, uri)
    device = torch.device(cfg.device)
    full_ref = quality_metrics(FULL_REFERENCE, cfg.device)
    no_ref = quality_metrics(NO_REFERENCE, cfg.device)

    tags = {"mode": "bench", "checkpoint": name}
    with tracking.start_run(cfg, parent_run_id=ckpt["run_id"], tags=tags):
        mlflow.log_metrics(_cost(model, device))
        for dataset in datasets:
            key = dataset.root.name
            mlflow.log_param(f"{key}.manifest", manifest(dataset.root))
            scores = _bench_dataset(
                model,
                dataset,
                full_ref,
                no_ref,
                device,
                cfg.train.precision,
                cfg.data.num_workers,
            )
            mlflow.log_metrics({f"{key}/{k}": v for k, v in scores.items()})
            print(
                f"[BENCH] {name} {key}: "
                + ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
            )


@torch.no_grad()
def _cost(model: nn.Module, device: torch.device) -> dict[str, float]:
    """Parameters and MACs at 256x256. FFT ops are not counted."""
    # FlopCounterMode misses attention on the eval-mode MultiheadAttention fast
    # path and in the CPU flash kernel (measured: ~30x undercount). Route MHA
    # through SDPA and count the CPU kernel like the CUDA ones. Forcing the math
    # backend instead would materialise a 65536^2 attention matrix.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    cpu_sdpa = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_flop}
    try:
        with FlopCounterMode(display=False, custom_mapping=cpu_sdpa) as counter:
            model(torch.zeros(1, 3, 256, 256, device=device))
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    return {
        "cost/params": sum(p.numel() for p in model.parameters()),
        "cost/macs_256": counter.get_total_flops() / 2,
    }


def _sdpa_flop(query, key, value, *args, out_shape=None, **kwargs) -> int:
    return sdpa_flop_count(query, key, value)


def _bench_dataset(
    model: nn.Module,
    dataset: LowLightDataset,
    full_ref: dict[str, nn.Module],
    no_ref: dict[str, nn.Module],
    device: torch.device,
    precision: str,
    num_workers: int,
) -> dict[str, float]:
    if not dataset.paired:
        raise ValueError(f"Benchmark dataset {dataset.root} has no high/ folder")
    sums = dict.fromkeys(FULL_REFERENCE + NO_REFERENCE, 0.0)
    times: list[float] = []
    peak = 0
    for i, batch in enumerate(make_loader(dataset, 1, False, num_workers)):
        normal = batch["normal"].to(device)
        enhanced, seconds, memory = _timed_forward(
            model, batch["low"].to(device), device, precision
        )
        if i > 0:  # the first image pays for kernel selection and allocation
            times.append(seconds)
        peak = max(peak, memory)
        for k, m in full_ref.items():
            sums[k] += m(enhanced, normal).item()
        for k, m in no_ref.items():
            sums[k] += m(enhanced).item()

    scores = {k: v / len(dataset) for k, v in sums.items()}
    scores["latency_ms"] = 1000 * (sum(times) / len(times) if times else seconds)
    if device.type == "cuda":
        weights = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        scores["peak_memory_mb"] = (peak + weights) / 2**20
    return scores


@torch.no_grad()
def _timed_forward(
    model: nn.Module, low: torch.Tensor, device: torch.device, precision: str
) -> tuple[torch.Tensor, float, int]:
    """Enhanced image, wall time, and peak memory the forward pass added."""
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with autocast(device, precision):
        enhanced = model(low)["enhanced"]["rgb"].float()
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    memory = torch.cuda.max_memory_allocated(device) - base if cuda else 0
    return enhanced, seconds, memory


@torch.no_grad()
def infer(cfg: DictConfig) -> None:
    """Enhanced images for the Inference set, saved as `<dataset>/<file>.png`
    artifacts of a child Run."""
    if not cfg.ckpt:
        raise ValueError("mode=infer needs ckpt=runs:/<run_id>/checkpoints/<name>.pt")
    datasets = build_datasets(cfg.data.infer_dir)
    if not datasets:
        raise FileNotFoundError(f"No Inference set in {cfg.data.infer_dir}")
    model, ckpt, name = load_checkpoint(cfg, cfg.ckpt)
    device = torch.device(cfg.device)

    tags = {"mode": "infer", "checkpoint": name}
    with (
        tracking.start_run(cfg, parent_run_id=ckpt["run_id"], tags=tags),
        tempfile.TemporaryDirectory() as tmp,
    ):
        for dataset in datasets:
            for batch in make_loader(dataset, 1, False, cfg.data.num_workers):
                with autocast(device, cfg.train.precision):
                    enhanced = model(batch["low"].to(device))["enhanced"]["rgb"].float()
                path = (
                    Path(tmp)
                    / dataset.root.name
                    / Path(batch["name"][0]).with_suffix(".png")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image(enhanced, path)
        mlflow.log_artifacts(tmp, "images")
        print(f"[INFO] Enhanced images: {mlflow.get_artifact_uri('images')}")
