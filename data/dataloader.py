from pathlib import Path
from typing import Literal, TypeAlias, Union, overload

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from data.utils import LowLightDataset, LowLightSample

LowLightDatasetList: TypeAlias = list[LowLightDataset]
LowLightDataLoader: TypeAlias = DataLoader[LowLightSample]
LowLightDataLoaderList: TypeAlias = list[LowLightDataLoader]


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
        self.train_dir: Path = Path(train_dir)
        self.valid_dir: Path = Path(valid_dir)
        self.bench_dir: Path = Path(bench_dir)
        self.infer_dir: Path = Path(infer_dir)

        self.image_size: int = image_size
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

        self.train_datasets: LowLightDatasetList = []
        self.valid_datasets: LowLightDatasetList = []
        self.bench_datasets: LowLightDatasetList = []
        self.infer_datasets: LowLightDatasetList = []

    def setup(
        self,
        stage: str,
    ) -> None:
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

    def _set_dataset(
        self,
        path: Path,
    ) -> LowLightDatasetList:
        datasets: LowLightDatasetList = []
        for folder in path.iterdir():
            if folder.is_dir():
                datasets.append(
                    LowLightDataset(
                        path=folder,
                        image_size=self.image_size,
                    )
                )
        return datasets

    @overload
    def _set_dataloader(
        self,
        datasets: LowLightDatasetList,
        concat: Literal[True],
        shuffle: bool = False,
    ) -> LowLightDataLoader: ...

    @overload
    def _set_dataloader(
        self,
        datasets: LowLightDatasetList,
        concat: Literal[False] = False,
        shuffle: bool = False,
    ) -> LowLightDataLoaderList: ...

    def _set_dataloader(
        self,
        datasets: LowLightDatasetList,
        concat: bool = False,
        shuffle: bool = False,
    ) -> Union[LowLightDataLoader, LowLightDataLoaderList]:
        if concat:
            dataset_concat: ConcatDataset[LowLightSample] = ConcatDataset(
                datasets=datasets,
            )
            dataloader: LowLightDataLoader = DataLoader(
                dataset=dataset_concat,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
            return dataloader
        dataloaders: LowLightDataLoaderList = []
        for dataset in datasets:
            loader: LowLightDataLoader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
            dataloaders.append(loader)
        return dataloaders

    def train_dataloader(self) -> LowLightDataLoader:
        return self._set_dataloader(
            datasets=self.train_datasets,
            concat=True,
            shuffle=True,
        )

    def val_dataloader(self) -> LowLightDataLoader:
        return self._set_dataloader(
            datasets=self.valid_datasets,
            concat=True,
        )

    def test_dataloader(self) -> LowLightDataLoaderList:
        return self._set_dataloader(
            datasets=self.bench_datasets,
        )

    def predict_dataloader(self) -> LowLightDataLoaderList:
        return self._set_dataloader(
            datasets=self.infer_datasets,
        )
