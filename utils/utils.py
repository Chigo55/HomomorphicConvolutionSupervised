import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from torchinfo import summary


def show_batch(images: torch.Tensor, ncols: int = 8):
    nimgs = images.shape[0]
    nrows = (nimgs + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 3, nrows * 3))
    for i in range(nimgs):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(X=F.to_pil_image(pic=images[i]))
        plt.axis('off')
        plt.title(label=f"Image {i}")
    plt.tight_layout()
    plt.show()


def make_dirs(path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_metrics(metrics: dict, prefix: str = ""):
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.4f}")


def save_images(results, save_dir, prefix="infer", ext="png"):
    for i, datasets in enumerate(iterable=results):
        save_path = make_dirs(path=f"{save_dir}/batch{i+1}")
        for ii, batch in enumerate(iterable=datasets):
            save_image(
                tensor=batch,
                fp=save_path / f"{prefix}_{ii:04d}.{ext}",
                nrow=8,
                padding=2,
                normalize=True,
                value_range=(0, 1)
            )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def summarize_model(model, input_size=None, input_data=None, **kwargs):
    if input_data is not None:
        return summary(model=model, input_data=input_data, **kwargs)
    elif input_size is not None:
        return summary(model=model, input_size=input_size, **kwargs)
    else:
        raise ValueError("Either input_data or input_size must be provided.")


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(tensor=m.weight)  # 또는 xavier_uniform_
        if m.bias is not None:
            nn.init.constant_(tensor=m.bias, val=0.0)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(tensor=m.weight, val=1.0)
        nn.init.constant_(tensor=m.bias, val=0.0)