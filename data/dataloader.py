from pathlib import Path
from typing import List, Literal, Optional, Union, overload

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from data.utils import LowLightDataset


class LowLightDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: Union[str, Path],
        valid_dir: Union[str, Path],
        bench_dir: Union[str, Path],
        infer_dir: Union[str, Path],
        image_size: int,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_dir = Path(train_dir)
        self.valid_dir = Path(valid_dir)
        self.bench_dir = Path(bench_dir)
        self.infer_dir = Path(infer_dir)

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None:
            self.train_datasets = self._set_dataset(path=self.train_dir)
            self.valid_datasets = self._set_dataset(path=self.valid_dir)
            self.bench_datasets = self._set_dataset(path=self.bench_dir)
            self.infer_datasets = self._set_dataset(path=self.infer_dir)
        elif stage == "fit":
            self.train_datasets = self._set_dataset(path=self.train_dir)
            self.valid_datasets = self._set_dataset(path=self.valid_dir)
        elif stage == "validate":
            self.valid_datasets = self._set_dataset(path=self.valid_dir)
        elif stage == "test":
            self.bench_datasets = self._set_dataset(path=self.bench_dir)
        elif stage == "predict":
            self.infer_datasets = self._set_dataset(path=self.infer_dir)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _set_dataset(self, path: Path) -> List[LowLightDataset]:
        datasets = []
        for folder in path.iterdir():
            if folder.is_dir():
                datasets.append(
                    LowLightDataset(path=folder, image_size=self.image_size)
                )
        return datasets

    @overload
    def _set_dataloader(
        self,
        datasets: List[LowLightDataset],
        concat: Literal[True],
        shuffle: bool = False,
    ) -> DataLoader: ...

    @overload
    def _set_dataloader(
        self,
        datasets: List[LowLightDataset],
        concat: Literal[False] = False,
        shuffle: bool = False,
    ) -> List[DataLoader]: ...

    def _set_dataloader(
        self,
        datasets: List[LowLightDataset],
        concat: bool = False,
        shuffle: bool = False,
    ) -> Union[DataLoader, List[DataLoader]]:
        if concat:
            dataloader = DataLoader(
                dataset=ConcatDataset(datasets=datasets),
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
            return dataloader
        else:
            dataloaders = []
            for dataset in datasets:
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                )
                dataloaders.append(loader)
            return dataloaders

    def train_dataloader(self) -> DataLoader:
        return self._set_dataloader(
            datasets=self.train_datasets, concat=True, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return self._set_dataloader(datasets=self.valid_datasets, concat=True)

    def test_dataloader(self) -> List[DataLoader]:
        return self._set_dataloader(datasets=self.bench_datasets)

    def predict_dataloader(self) -> List[DataLoader]:
        return self._set_dataloader(datasets=self.infer_datasets)