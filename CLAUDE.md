# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Low-light image enhancement research: plain PyTorch, Hydra config, MLflow tracking. Read `GLOSSARY.md` first and use its terms (Run, Experiment, Checkpoint, Benchmark result, Low-light/Normal-light/Enhanced image).

## Commands

```bash
uv sync                                   # Python 3.10, torch cu118; uv.lock is tracked
uv run python main.py                     # mode=fit (default), then bench of best.pt
uv run python main.py mode=bench ckpt=runs:/<run_id>/checkpoints/best.pt
uv run python main.py mode=infer ckpt=runs:/<run_id>/checkpoints/best.pt
uv run python main.py resume=runs:/<run_id>/checkpoints/last.pt
uv run pytest                             # CPU smoke tests, ~20 s
uv run ruff check . && uv run ruff format .
```

Training runs on a shared lab GPU server over SSH, not on the dev PC. The dev PC has no GPU, so pass `device=cpu train.precision=fp32` for local checks.

## Structure

- `main.py` chdirs to the repo (config paths are repo-relative), then dispatches `cfg.mode` to `engine.fit.fit`, `engine.evaluate.bench` or `engine.evaluate.infer`.
- `conf/config.yaml` is the single config. Hydra's output dir is disabled (`hydra.run.dir: .`); every Run output goes to MLflow artifacts. Experiments are `conf/experiment/*.yaml` files with `# @package _global_` that set `experiment_name` plus overrides.
- `engine/tracking.py`: MLflow is SQLite (`mlflow.db`) plus local artifacts (`mlartifacts/<experiment>/`), written directly with no server. `start_run` tees stdout/stderr into `logs/<UTC time>.log` for every Run. bench and infer are child Runs of the fit Run whose id is stored in the checkpoint.
- Checkpoints are `torch.save` dicts: `model, optimizer, scheduler, scaler, state, rng, config, run_id`, loaded only through `runs:/` URIs. `load_checkpoint` rebuilds the model from `ckpt["config"]["model"]`, so architecture overrides on the CLI are ignored for bench and infer.
- `resume` uses the checkpoint's config. Only `RESUME_OVERRIDES` in `engine/fit.py` can change it, and only when the CLI value differs from `conf/config.yaml`. RNG state is restored right before the epoch loop. `tests/test_smoke.py` checks that a resumed epoch replays exactly.
- `data/dataset.py` items are dicts `{low_light, normal_light?, name}`. `normal_light` is present only when a `high/` folder exists.
- `engine.evaluate.enhance()` is the one forward path: autocast at `train.precision`, float32 output. Losses run outside autocast.

## Model pipeline (model/blocks/lowlightenhancer.py)

1. `ImageDecomposition`: RGB → YCrCb (Cr and Cb shifted by `offset`), then a homomorphic split of Y: log → FFT → Gaussian LPF (`cutoff`) → illumination `il` and reflectance `re`.
2. `FeatureRestorationBlock`: Cr goes through `cr_conv` and Cb through `cb_conv` (ResidualBlock → SelfAttention → ResidualBlock).
3. `IlluminationEnhancer`: a U-Net with `CrossAttentionBlock`. The query is `il` and the key/value is `re`. One shared `in_conv` serves both streams. Output is `out_conv(x) + il * re`.
4. `ImageComposition`: YCrCb → RGB, clamped to [0, 1].

`forward` returns `{"low_light": {...}, "enhanced": {...}}`. `notebooks/figures.ipynb` reads those keys.

Attention runs over every H×W token at full resolution, which makes it memory-heavy. Validation, bench and infer use full images with batch size 1, so H and W must be divisible by `2 ** num_resolution`.

## Gotchas

- Metric keys: `train/loss/{mae,mse,ssim,total}`, `valid/loss/*`, `valid/psnr`, `valid/ssim`. `train.monitor` must be one of the `valid/*` keys; a typo fails after the first epoch.
- `cost/macs_256` disables the MHA fast path while counting and maps the CPU flash-attention op to the SDPA formula. Without that, `FlopCounterMode` drops attention (~30x undercount). Don't force the math SDPA backend: at 256² it materialises a 65536² matrix.
- pyiqa downloads LPIPS/NIQE/BRISQUE weights on first bench (internet needed).
- Notifications: `DISCORD_WEBHOOK_URL` env var. Never put the URL in config or git.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (Chigo55/HomomorphicConvolutionSupervised) via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/` (created lazily). See `docs/agents/domain.md`.
