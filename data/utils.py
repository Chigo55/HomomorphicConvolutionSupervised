import random
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class LowLightDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        path: str | Path,
        image_size: int,
        augment: bool,
        crop: bool,
    ) -> None:
        super().__init__()
        self.path: Path = Path(path)
        self.image_size: int = image_size
        self.augment: bool = augment
        self.crop: bool = crop

        self.low_path: Path = self.path / "low"
        self.high_path: Path = self.path / "high"

        self.low_datas: list[Path] = sorted(self.low_path.rglob(pattern="*.*"))
        self.high_datas: list[Path] = sorted(self.high_path.rglob(pattern="*.*"))

    def __len__(self) -> int:
        return len(self.low_datas)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        low_data: Path = self.low_datas[index]
        high_data: Path = self.high_path / low_data.name

        low_image: Image.Image = Image.open(fp=low_data).convert(mode="RGB")
        high_image: Image.Image = Image.open(fp=high_data).convert(mode="RGB")

        if self.augment:
            low_image, high_image = self._pair_augment(
                low=low_image,
                high=high_image,
            )

        if self.crop:
            low_image, high_image = self._pair_random_crop(
                low=low_image,
                high=high_image,
                patch_size=self.image_size,
            )

        return F.to_tensor(pic=low_image), F.to_tensor(pic=high_image)

    def _pair_augment(
        self,
        low: Image.Image,
        high: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            low = F.hflip(img=low)  # type: ignore
            high = F.hflip(img=high)  # type: ignore

        if random.random() < 0.5:
            low = F.vflip(img=low)  # type: ignore
            high = F.vflip(img=high)  # type: ignore

        return low, high

    def _pair_random_crop(
        self,
        low: Image.Image,
        high: Image.Image,
        patch_size: int,
    ) -> tuple[Image.Image, Image.Image]:
        w, h = low.size

        if w < patch_size or h < patch_size:
            low = F.resize(
                img=low,  # type: ignore
                size=[patch_size, patch_size],
                interpolation=F.InterpolationMode.BICUBIC,
            )
            high = F.resize(
                img=high,  # type: ignore
                size=[patch_size, patch_size],
                interpolation=F.InterpolationMode.BICUBIC,
            )
            return low, high

        if w == patch_size and h == patch_size:
            return low, high

        left = random.randint(a=0, b=w - patch_size)
        top = random.randint(a=0, b=h - patch_size)

        low = F.crop(
            img=low,  # type: ignore
            top=top,
            left=left,
            height=patch_size,
            width=patch_size,
        )
        high = F.crop(
            img=high,  # type: ignore
            top=top,
            left=left,
            height=patch_size,
            width=patch_size,
        )

        return low, high
