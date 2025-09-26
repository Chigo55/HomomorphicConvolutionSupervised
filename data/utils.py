from pathlib import Path
from typing import Union, cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LowLightDataset(Dataset):
    def __init__(self, path: Union[str, Path], image_size: int) -> None:
        super().__init__()
        self.path = Path(path)
        self.image_size = image_size
        self.transform = transforms.Compose(
            transforms=[
                transforms.Resize(size=(self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.low_path = self.path / "low"
        self.high_path = self.path / "high"

        self.low_datas = sorted(list(self.low_path.rglob(pattern="*.*")))
        self.high_datas = sorted(list(self.high_path.rglob(pattern="*.*")))

    def __len__(self) -> int:
        return len(self.low_datas)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        low_data = self.low_datas[index]
        high_data = self.high_path / low_data.name

        low_image = Image.open(fp=low_data).convert(mode="RGB")
        high_image = Image.open(fp=high_data).convert(mode="RGB")

        low_data_tensor = cast(torch.Tensor, self.transform(img=low_image))
        high_data_tensor = cast(torch.Tensor, self.transform(img=high_image))

        return low_data_tensor, high_data_tensor
