# Homomorphic Convolution Supervised

Supervised low-light image enhancement. The model splits luminance into illumination and reflectance with a homomorphic filter, enhances illumination with a cross-attention U-Net, restores chroma, and recomposes RGB. Training is plain PyTorch, configuration is Hydra, and every Run is tracked in MLflow. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).

## Setup

```bash
uv sync    # Python 3.10, torch 2.7.1+cu118
```

## Data

Each split holds one folder per dataset, with `low/` and `high/` images paired by relative path:

```
data/
├── 1_train/<dataset>/{low,high}/   # Training set (random crops + flips)
├── 2_valid/<dataset>/{low,high}/   # Validation set (picks the best Checkpoint)
├── 3_bench/<dataset>/{low,high}/   # Benchmark set (reported per dataset)
└── 4_infer/<dataset>/low/          # Inference set (high/ optional, never scored)
```

Validation, bench and infer run on full images, so H and W must be divisible by `2 ** model.num_resolution`.

## Running

```bash
# Baseline Run: train, then benchmark the best Checkpoint
uv run python main.py

# Any config value can be overridden
uv run python main.py train.lr=3e-4 model.cutoff=0.2

# An Experiment is a file in conf/experiment/ holding only what differs
uv run python main.py +experiment=cutoff_ablation -m model.cutoff=0.05,0.1,0.2

# Benchmark or enhance with any Checkpoint (run id from the MLflow UI)
uv run python main.py mode=bench ckpt=runs:/<run_id>/checkpoints/epoch-020.pt
uv run python main.py mode=infer ckpt=runs:/<run_id>/checkpoints/best.pt

# Resume a crashed Run (keeps its original config)
uv run python main.py resume=runs:/<run_id>/checkpoints/last.pt
```

Settings live in [conf/config.yaml](conf/config.yaml). The ones you'll change most often:

| Key | Default | Meaning |
| --- | --- | --- |
| `experiment_name` | `baseline` | MLflow Experiment the Run goes into |
| `seed` | `42` | Fixed. Use 42/43/44 for Runs in the final table |
| `train.monitor` / `train.monitor_mode` | `valid/psnr` / `max` | Picks `best.pt` and drives early stopping |
| `train.precision` | `fp16` | `fp16`, `bf16` or `fp32` |
| `device` | `cuda` | `cpu` works for smoke checks |

## Tracking (MLflow)

Runs are written straight to `mlflow.db` and `mlartifacts/` in the repo, so training doesn't need a server. On the training server:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

From your own machine: `ssh -L 5000:127.0.0.1:5000 <server>`, then open http://localhost:5000. Keep `--host 127.0.0.1`: the MLflow UI has no authentication.

A fit Run records:
- the resolved `config.yaml`, every config value as a param, data manifests, the git commit, and `uncommitted.patch` if the tree was dirty
- `train/loss/*`, `train/lr` and `train/grad_norm` every `train.log_every_n_steps` steps; `valid/loss/*`, `valid/psnr` and `valid/ssim` every epoch, averaged over the whole Validation set
- `valid/sample_*` images (low-light | enhanced | normal-light) for the same validation images every epoch
- GPU, CPU and memory system metrics
- `checkpoints/best.pt`, `last.pt` and `epoch-XXX.pt` every `train.checkpoint_every` epochs

Each bench or infer call is a child Run of its fit Run, tagged with `mode` and `checkpoint`. Bench logs `<dataset>/{psnr,ssim,lpips,niqe,brisque,latency_ms}` and, on CUDA, `<dataset>/peak_memory_mb`, plus `cost/params` and `cost/macs_256`. Infer stores `images/<dataset>/<file>.png`.

Training stops right away on a non-finite loss. Set `DISCORD_WEBHOOK_URL` to get a message when a Run finishes or crashes.

## Layout

```
main.py                 Hydra entry point: mode=fit|bench|infer
conf/                   config.yaml, experiment/*.yaml
data/dataset.py         paired dataset, loaders, split manifest
engine/fit.py           training loop, checkpoints, early stopping
engine/evaluate.py      validation, bench (quality + cost), infer
engine/tracking.py      MLflow runs, checkpoint URIs, notifications
model/blocks/           the network
model/loss.py           MAE + MSE + (1 - SSIM)
notebooks/figures.ipynb paper figures from a Checkpoint
tests/test_smoke.py     CPU smoke tests
```

## Checks

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```
