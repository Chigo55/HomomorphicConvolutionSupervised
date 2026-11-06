import hashlib
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class LowLightDataset(Dataset):
    """One dataset folder: `low/` Low-light images, optional `high/` Normal-light
    images paired by relative path. Items are dicts with `low_light`, `name` and,
    when paired, `normal_light`."""

    def __init__(
        self, root: str | Path, crop_size: int | None = None, augment: bool = False
    ) -> None:
        self.root = Path(root)
        self.low_dir = self.root / "low"
        self.normal_dir = self.root / "high"
        self.paired = self.normal_dir.is_dir()
        self.files = sorted(
            p.relative_to(self.low_dir)
            for p in self.low_dir.rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        rel = self.files[index]
        images = [Image.open(self.low_dir / rel).convert("RGB")]
        if self.paired:
            images.append(Image.open(self.normal_dir / rel).convert("RGB"))

        if self.augment:
            if random.random() < 0.5:
                images = [F.hflip(im) for im in images]
            if random.random() < 0.5:
                images = [F.vflip(im) for im in images]
        if self.crop_size:
            images = _random_crop(images, self.crop_size)

        item = {"low_light": F.to_tensor(images[0]), "name": rel.as_posix()}
        if self.paired:
            item["normal_light"] = F.to_tensor(images[1])
        return item


def _random_crop(images: list[Image.Image], size: int) -> list[Image.Image]:
    w, h = images[0].size
    if w < size or h < size:
        return [
            F.resize(im, [size, size], interpolation=F.InterpolationMode.BICUBIC)
            for im in images
        ]
    left = random.randint(0, w - size)
    top = random.randint(0, h - size)
    return [F.crop(im, top, left, size, size) for im in images]


def build_datasets(split_dir: str | Path, **kwargs) -> list[LowLightDataset]:
    """One LowLightDataset per dataset folder in a split, sorted by name."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    return [
        LowLightDataset(d, **kwargs) for d in sorted(split_dir.iterdir()) if d.is_dir()
    ]


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def manifest(split_dir: str | Path) -> str:
    """Short hash of every file's relative path and size in a split."""
    # ponytail: path+size, not content; content hashing would re-read GBs every Run
    split_dir = Path(split_dir)
    lines = sorted(
        f"{p.relative_to(split_dir).as_posix()}:{p.stat().st_size}"
        for p in split_dir.rglob("*")
        if p.is_file()
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]
