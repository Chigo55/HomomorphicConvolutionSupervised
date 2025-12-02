from pathlib import Path

import lightning as L
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

from data.utils import LowLightDataset


class LowLightDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        bench_dir: str,
        infer_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dir: Path = Path(train_dir)
        self.valid_dir: Path = Path(valid_dir)
        self.bench_dir: Path = Path(bench_dir)
        self.infer_dir: Path = Path(infer_dir)

        self.image_size: int = image_size
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_datasets: list[LowLightDataset] = []
        self.valid_datasets: list[LowLightDataset] = []
        self.bench_datasets: list[LowLightDataset] = []
        self.infer_datasets: list[LowLightDataset] = []

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_datasets = self._set_dataset(
                data_dir=self.train_dir, augment=True, crop=True
            )

        if stage in (None, "fit", "validate"):
            self.valid_datasets = self._set_dataset(
                data_dir=self.valid_dir, augment=False, crop=False
            )

        if stage in (None, "test"):
            self.bench_datasets = self._set_dataset(
                data_dir=self.bench_dir, augment=False, crop=False
            )

        if stage in (None, "predict"):
            self.infer_datasets = self._set_dataset(
                data_dir=self.infer_dir, augment=False, crop=False
            )

    def _set_dataset(
        self,
        data_dir: Path,
        augment: bool,
        crop: bool,
    ) -> list[LowLightDataset]:
        if not data_dir.exists():
            return []
        else:
            return [
                LowLightDataset(
                    path=folder,
                    image_size=self.image_size,
                    augment=augment,
                    crop=crop,
                )
                for folder in sorted(data_dir.iterdir())
                if folder.is_dir()
            ]

    def _set_dataloader(
        self,
        datasets: list[LowLightDataset],
        concat: bool = True,
        shuffle: bool = True,
        batch_size: int | None = None,
    ) -> DataLoader[tuple[Tensor, Tensor]] | list[DataLoader[tuple[Tensor, Tensor]]]:
        batch_size = batch_size if batch_size is not None else self.batch_size

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0,
            "pin_memory": True,
        }

        if not datasets:
            return []

        if concat:
            return DataLoader(dataset=ConcatDataset(datasets=datasets), **loader_kwargs)
        else:
            return [DataLoader(dataset=ds, **loader_kwargs) for ds in datasets]

    def train_dataloader(
        self,
        concat: bool = True,
        shuffle: bool = True,
    ) -> DataLoader[tuple[Tensor, Tensor]] | list[DataLoader[tuple[Tensor, Tensor]]]:
        return self._set_dataloader(
            datasets=self.train_datasets,
            concat=concat,
            shuffle=shuffle,
        )

    def val_dataloader(
        self,
        concat: bool = True,
        shuffle: bool = False,
    ) -> DataLoader[tuple[Tensor, Tensor]] | list[DataLoader[tuple[Tensor, Tensor]]]:
        return self._set_dataloader(
            datasets=self.valid_datasets,
            concat=concat,
            shuffle=shuffle,
            batch_size=1
        )

    def test_dataloader(
        self,
        concat: bool = False,
        shuffle: bool = False,
    ) -> DataLoader[tuple[Tensor, Tensor]] | list[DataLoader[tuple[Tensor, Tensor]]]:
        return self._set_dataloader(
            datasets=self.bench_datasets,
            concat=concat,
            shuffle=shuffle,
            batch_size=1
        )

    def predict_dataloader(
        self,
        concat: bool = False,
        shuffle: bool = False,
    ) -> DataLoader[tuple[Tensor, Tensor]] | list[DataLoader[tuple[Tensor, Tensor]]]:
        return self._set_dataloader(
            datasets=self.infer_datasets,
            concat=concat,
            shuffle=shuffle,
            batch_size=1
        )
