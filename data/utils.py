from pathlib import Path
from typing import TypeAlias, Union, cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LowLightSample: TypeAlias = tuple[torch.Tensor, torch.Tensor]


class LowLightDataset(Dataset[LowLightSample]):
    def __init__(self, path: Union[str, Path], image_size: int) -> None:
        super().__init__()
        self.path: Path = Path(path)
        self.image_size: int = image_size
        self.transform: transforms.Compose = transforms.Compose(
            transforms=[
                transforms.Resize(size=(self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.low_path: Path = self.path / "low"
        self.high_path: Path = self.path / "high"

        self.low_datas: list[Path] = sorted(list(self.low_path.rglob(pattern="*.*")))
        self.high_datas: list[Path] = sorted(list(self.high_path.rglob(pattern="*.*")))

    def __len__(self) -> int:
        return len(self.low_datas)

    def __getitem__(self, index: int) -> LowLightSample:
        low_data: Path = self.low_datas[index]
        high_data: Path = self.high_path / low_data.name

        low_image: Image.Image = Image.open(fp=low_data).convert(mode="RGB")
        high_image: Image.Image = Image.open(fp=high_data).convert(mode="RGB")

        low_data_tensor: torch.Tensor = cast(
            torch.Tensor, self.transform(img=low_image)
        )
        high_data_tensor: torch.Tensor = cast(
            torch.Tensor, self.transform(img=high_image)
        )

        return low_data_tensor, high_data_tensor
