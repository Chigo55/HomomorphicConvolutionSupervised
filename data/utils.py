import random
from pathlib import Path
from typing import Tuple, cast

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

LowLightSample = Tuple[Tensor, Tensor]


class LowLightDataset(Dataset[LowLightSample]):
    def __init__(
        self,
        path: str | Path,
        image_size: int,
        augment: bool,
    ) -> None:
        super().__init__()
        self.path: Path = Path(path)
        self.image_size: int = image_size
        self.augment: bool = augment

        self.transform: transforms.Compose = transforms.Compose(
            transforms=[
                transforms.ToTensor(),
            ]
        )

        self.low_path: Path = self.path / "low"
        self.high_path: Path = self.path / "high"

        self.low_datas: list[Path] = sorted(self.low_path.rglob(pattern="*.*"))
        self.high_datas: list[Path] = sorted(self.high_path.rglob(pattern="*.*"))

    def __len__(self) -> int:
        return len(self.low_datas)

    def __getitem__(self, index: int) -> LowLightSample:
        low_data: Path = self.low_datas[index]
        high_data: Path = self.high_path / low_data.name

        low_image: Image.Image = Image.open(fp=low_data).convert(mode="RGB")
        high_image: Image.Image = Image.open(fp=high_data).convert(mode="RGB")

        if self.augment:
            low_image, high_image = self._pair_augment(
                low_image=low_image,
                high_image=high_image,
            )

        low_image, high_image = self._pair_random_crop(
            low_image=low_image,
            high_image=high_image,
            patch_size=self.image_size,
        )

        low_tensor: Tensor = cast(Tensor, self.transform(img=low_image))
        high_tensor: Tensor = cast(Tensor, self.transform(img=high_image))

        return low_tensor, high_tensor

    def _pair_augment(
        self,
        low_image: Image.Image,
        high_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            low_image = low_image.transpose(method=Image.FLIP_LEFT_RIGHT)
            high_image = high_image.transpose(method=Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            low_image = low_image.transpose(method=Image.FLIP_TOP_BOTTOM)
            high_image = high_image.transpose(method=Image.FLIP_TOP_BOTTOM)

        return low_image, high_image

    def _pair_random_crop(
        self,
        low_image: Image.Image,
        high_image: Image.Image,
        patch_size: int,
    ) -> tuple[Image.Image, Image.Image]:
        assert low_image.size == high_image.size, "low/high 이미지 크기가 다릅니다."

        w, h = low_image.size

        if w < patch_size or h < patch_size:
            low_image = low_image.resize(size=(patch_size, patch_size), resample=Image.BICUBIC)
            high_image = high_image.resize(size=(patch_size, patch_size), resample=Image.BICUBIC)
            return low_image, high_image

        if w == patch_size and h == patch_size:
            return low_image, high_image

        left = random.randint(a=0, b=w - patch_size)
        top = random.randint(a=0, b=h - patch_size)
        right = left + patch_size
        bottom = top + patch_size

        low_crop = low_image.crop(box=(left, top, right, bottom))
        high_crop = high_image.crop(box=(left, top, right, bottom))

        return low_crop, high_crop
