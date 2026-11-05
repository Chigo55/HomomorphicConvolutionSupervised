# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Low-light image enhancement (PyTorch Lightning). README.md is partly stale: it refers to a `main.py` that no longer exists.

## Commands

```bash
uv sync                    # canonical env: Python 3.10, torch cu118 (pyproject.toml); requirements.txt is stale
uv run python train.py     # fit, then bench on data/3_bench
uv run python valid.py     # validate a checkpoint
uv run python bench.py     # PSNR/SSIM/LPIPS/NIQE/BRISQUE on data/3_bench
uv run python infer.py     # save enhanced images
uv run ruff check . && uv run ruff format .
tensorboard --logdir runs/
```

There is no pytest suite. `test/*.ipynb` are notebooks that exercise each block one at a time.

## Entry scripts

`train.py`, `valid.py`, `bench.py` and `infer.py` each contain their own copy of `get_hparams()`, and the eval scripts hard-code `checkpoint_path`. A change to an hparam has to be made in all four. `train.py` replaces `seed` with a random value on every run.

`LightningEngine` (engine/engine.py) builds one model, logger, trainer and callback set. Its `train/valid/bench/infer` methods wrap Lightning's `fit/validate/test/predict` through thin runners in engine/runner.py, and each runner builds its own `LowLightDataModule`.

`load_from_checkpoint(..., hparams=self.hparams)` passes the current script's hparams in place of the saved ones. Architecture keys (`embed_dim`, `num_heads`, `mlp_ratio`, `num_resolution`) must match the checkpoint, or loading fails.

## Model pipeline (model/blocks/lowlightenhancer.py)

1. `ImageDecomposition`: RGB → YCrCb (Cr and Cb shifted by `offset`), then a homomorphic split of Y: log → FFT → Gaussian LPF (`cutoff`) → illumination `il` and reflectance `re`.
2. `FeatureRestorationBlock`: Cr and Cb each go through ResidualBlock → SelfAttention → ResidualBlock. Note that `forward` currently uses `cr_conv` for both channels.
3. `IlluminationEnhancer`: a U-Net with `CrossAttentionBlock`. The query is `il` and the key/value is `re`. One shared `in_conv` serves both streams. Output is `out_conv(x) + il * re`.
4. `ImageComposition`: YCrCb → RGB, clamped to [0, 1].

`forward` returns `{"low": {...}, "enhanced": {...}}`. The image logging in model/model.py iterates over those dicts, so changing their keys changes the TensorBoard tag names.

Attention runs over every H×W token at full resolution, which makes it memory-heavy. Valid, bench and infer run on full, uncropped images with batch_size 1. H and W must be divisible by `2 ** num_resolution`.

## Gotchas

- Metric names are built as `f"{stage}/{i+1}_{key}"` from the order of the loss dict (mae, mse, ssim, total). The total loss is therefore `valid/4_total`. The eval scripts set `"valid/3_total"`, which is stale, but `monitor_metric` only takes effect during fit.
- `pyiqa` metrics and the SSIM loss hard-code `device="cuda"`, so a GPU is required.
- `transformers` (cosine warmup scheduler) and `matplotlib` (utils/utils.py) are imported but not declared in pyproject.toml.
- Data layout: `data/{1_train,2_valid,3_bench,4_infer}/<dataset>/{low,high}/`. Low and high images are paired by filename. Each split loads every dataset folder; train and valid concatenate them, while bench and infer keep one dataloader per dataset.
- Outputs go to `runs/<experiment_name>/version_N/`: checkpoints `best.ckpt`, `last.ckpt` and `epoch-XX.ckpt` every 5 epochs, plus `inference/dataset_XX/` for inference images.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (Chigo55/HomomorphicConvolutionSupervised) via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/` (created lazily). See `docs/agents/domain.md`.
