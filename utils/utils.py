from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchinfo import summary
from torchvision.utils import save_image

PathLike: TypeAlias = str | Path
TensorBatch: TypeAlias = Iterable[torch.Tensor]
TensorBatches: TypeAlias = Iterable[TensorBatch]
MetricMapping: TypeAlias = Mapping[str, float]


def show_batch(images: torch.Tensor, ncols: int = 8) -> None:
    nimgs: int = images.shape[0]
    nrows: int = (nimgs + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 3, nrows * 3))
    for i in range(nimgs):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(X=F.to_pil_image(pic=images[i]))
        plt.axis("off")
        plt.title(label=f"Image {i}")
    plt.tight_layout()
    plt.show()


def make_dirs(path: PathLike) -> Path:
    path_obj: Path = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def print_metrics(metrics: MetricMapping, prefix: str = "") -> None:
    for key, value in metrics.items():
        print(f"{prefix}{key}: {value:.4f}")


def save_images(
    results: TensorBatches,
    save_dir: PathLike,
    prefix: str = "infer",
    ext: str = "png",
) -> None:
    for i, datasets in enumerate(iterable=results):
        save_path: Path = make_dirs(path=f"{save_dir}/batch{i + 1}")
        for ii, batch in enumerate(iterable=datasets):
            save_image(
                tensor=batch,
                fp=save_path / f"{prefix}_{ii:04d}.{ext}",
                nrow=8,
                padding=2,
                normalize=True,
                value_range=(0, 1),
            )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize_model(
    model: nn.Module,
    input_size: Sequence[int] | Sequence[Sequence[int]] | None = None,
    input_data: Any = None,
    **kwargs: Any,
) -> Any:
    if input_data is not None:
        return summary(model=model, input_data=input_data, **kwargs)
    if input_size is not None:
        return summary(model=model, input_size=input_size, **kwargs)
    raise ValueError("Either input_data or input_size must be provided.")


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(tensor=m.weight)
        if m.bias is not None:
            nn.init.constant_(tensor=m.bias, val=0.0)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(tensor=m.weight, val=1.0)
        nn.init.constant_(tensor=m.bias, val=0.0)