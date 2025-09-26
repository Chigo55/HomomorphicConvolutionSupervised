# Homomorphic Convolution Supervised

Low-light image enhancement pipeline that decomposes illumination with homomorphic filtering, restores chroma and reflectance details, and re-composes enhanced RGB outputs. The implementation uses PyTorch Lightning for training, validation, benchmarking, and inference.

## Repository Layout

```
.
├── data/
│   ├── __init__.py
│   ├── dataloader.py
│   └── utils.py
├── engine/
│   ├── __init__.py
│   ├── engine.py
│   └── runner.py
├── model/
│   ├── __init__.py
│   ├── model.py
│   ├── loss.py
│   └── blocks/
│       ├── __init__.py
│       ├── featurerestorer.py
│       ├── homomorphic.py
│       ├── illuminationenhancer.py
│       └── lowlightenhancer.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── utils.py
├── main.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Environment Management (UV)

This project uses [UV](https://github.com/astral-sh/uv) for Python environment and dependency management.

```bash
# create and sync the environment
uv sync

# run commands inside the environment
uv run python main.py

# add or update dependencies
uv add <package>
uv lock
```

Requirements remain in
`requirements.txt` for compatibility with other workflows, but `uv sync` is the canonical way to reproduce the environment described in `pyproject.toml` and `uv.lock`.

## Architecture Overview

| Stage | File | Description |
| --- | --- | --- |
| Data loading | data/utils.py, data/dataloader.py | LowLightDataset pairs low and high images; LowLightDataModule builds loaders for train, valid, bench, infer splits. |
| Homomorphic separation | model/blocks/homomorphic.py | Converts RGB and YCrCb and splits illumination and reflectance via a learned homomorphic filter. |
| Illumination enhancement | model/blocks/illuminationenhancer.py | U-Net style convolutional stack that brightens the illumination map. |
| Feature restoration | model/blocks/featurerestorer.py | Restores chroma and reflectance through residual double-convolution blocks. |
| Recomposition | model/blocks/lowlightenhancer.py | Combines restored components, clamps outputs, and returns structured tensors. |
| Lightning wrapper | model/model.py | Aggregates the pipeline, computes MAE, MSE, SSIM losses, logs metrics, and configures optimisers. |
| Engine and runners | engine/engine.py, engine/runner.py | LightningEngine seeds, logs, and holds a single model instance; stage runners reuse that instance so validation, benchmarking, and inference use trained weights. |
| Metrics and utilities | utils/metrics.py, utils/utils.py | Quality metrics (PSNR, SSIM, LPIPS, NIQE, BRISQUE) and helpers for saving images, printing metrics, and summarising models. |

## Dataset Expectation

Each split directory can contain multiple datasets. Inside each dataset folder you must provide matching low/ and high/ subfolders. Example layout:

```
data/
├── 1_train/
│   └── LOLv1/
│       ├── low/
│       └── high/
├── 2_valid/
│   └── LOLv1/
│       ├── low/
│       └── high/
├── 3_bench/
│   └── LOLv1/
│       ├── low/
│       └── high/
└── 4_infer/
    └── LOLv1/
        ├── low/
        └── high/
```

File names inside corresponding low/ and high/ directories must align (for example 0001.png should exist in both).

## Running the Pipeline

```python
from engine import LightningEngine
from model.model import LowLightEnhancerLightning
from main import get_hparams

engine = LightningEngine(
    model_class=LowLightEnhancerLightning,
    hparams=get_hparams(),
)
engine.train()   # fits the model
engine.valid()   # evaluates on validation data
engine.bench()   # reports PSNR, SSIM, LPIPS, NIQE, BRISQUE
engine.infer()   # saves enhanced images for inference data
```

### Resuming or Inference from a Checkpoint

```python
engine = LightningEngine(
    model_class=LowLightEnhancerLightning,
    hparams=get_hparams(),
    checkpoint_path="runs/example/best-epoch.ckpt",
)
```

## Hyperparameters

main.py exposes get_hparams(); it returns a dictionary that contains everything the engine runners need. The default keys cover:

- **Data**: 	train_data_path, valid_data_path, bench_data_path, infer_data_path, image_size, batch_size,
um_workers.
- **Training loop**: seed, max_epochs, accelerator, devices, precision, log_every_n_steps, log_dir, experiment_name, inference (subdirectory name for predictions), patience (early stopping).
- **Model configuration**: hidden_channels,
um_resolution, dropout_ratio,
aw_cutoff, offset, 	rainable (whether to train homomorphic filters / down-sampling convs).
- **Optimiser**: optim (string name). Optional keys such as lr, momentum, betas, etc., are consumed inside configure_optimizers depending on which optimiser you choose.

The sample script loops over several optimiser options to compare performance. If you only need one configuration, remove the loop and set the desired optimiser and hyperparameters directly in get_hparams() or via CLI/environment overrides.

## Metrics and Logging

- Metrics are computed through utils/metrics.py and logged by Lightning.
- TensorBoard logs and checkpoints are stored under log_dir/experiment_name (defaults to runs/<experiment>).
- The best checkpoint is tracked using the validation total loss (valid/4_total).

## Notes

- Designed for research and experimentation; additional robustness checks are recommended before production deployment.
- Utility helpers (utils/utils.py) cover image saving, metric printing, directory creation, parameter counting, and model summaries.
- Loss wrappers in model/loss.py can be extended for custom objectives.