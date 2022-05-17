from abc import ABC, abstractmethod, abstractproperty
from ast import Sub
from dataclasses import dataclass
from functools import partial
import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from eyemind.dataloading.load_dataset import get_filenames_for_dataset, get_label_df, get_label_mapper, get_stratified_group_splits, limit_sequence_len

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler, Subset

from pytorch_lightning import LightningDataModule

from eyemind.dataloading.transforms import LimitSequenceLength, ToTensor


class SequenceToLabelDataset(Dataset):
    def __init__(self, folder_name, file_list=[], file_mapper=None, file_type="csv", transform_x=None, transform_y=None, label_mapper=None, skiprows=1, usecols=[1,2]):
        '''
        Dataset for large data with multiple csv files.

        Args:
        folder_name (str): path to folder where csv files are for x data
        file_list (list[str]): list of filenames to use for dataset
        file_mapper (fn): Gives list of files in folder to use for the dataset. Returns filename as string
        file_type (str): file extension. This is only used if file_mapper is none
        transform_x (list[fns]): functions that are applied to the x_data
        label_file (str): path to file specifying labels for each file
        label_mapper (fn): maps list of files to labels. Returns list
        '''

        self.folder_name = Path(folder_name)
        self.skiprows = skiprows
        self.usecols = usecols
        # If their is a list passed then use it, else if function then use it, else use all files in folder
        if file_list:
            self.files = file_list
        elif file_mapper:
            self.files = file_mapper(str(self.folder_name.resolve()))
        else:
            self.files = [str(f.resolve()) for f in self.folder_name.glob(f"*.{file_type}")]

        self.transform_x = transform_x
        self.transform_y = transform_y
        if label_mapper:
            self.labels = label_mapper(self.files)
        
    def __len__(self):
        return len(self.files)
    
    def _get_file_path(self,filename):
        return str(Path(self.folder_name,filename).resolve())
    
    def __getitem__(self,idx):
        filename = self.files[idx]
        filepath = self._get_file_path(filename)
        x_data = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=self.skiprows,usecols=self.usecols)
        if self.transform_x:
            x_data = self.transform_x(x_data)
        if self.labels:
            label=self.labels[idx]
            if self.transform_y:
                label = self.transform_y(label)
            return x_data, label
        else:
            return x_data

    def get_labels_from_indices(self, indices):
        return [self.labels[ind] for ind in indices]
    
    def get_files_from_indices(self, indices):
        return [self.files[ind] for ind in indices]

class MultiFileDataset(Dataset):
    def __init__(self, folder_name, file_list=[], file_mapper=None, file_type="csv", transform_x=None, transform_y=None, label_mapper=None):
        '''
        Dataset for large data with multiple csv files.

        Args:
        folder_name (str): path to folder where csv files are for x data
        file_list (list[str]): list of filenames to use for dataset
        file_mapper (fn): Gives list of files in folder to use for the dataset. Returns filename as string
        file_type (str): file extension. This is only used if file_mapper is none
        transform_x (list[fns]): functions that are applied to the x_data
        label_file (str): path to file specifying labels for each file
        label_mapper (fn): maps list of files to labels. Returns list
        '''

        self.folder_name = Path(folder_name)
        
        # If their is a list passed then use it, else if function then use it, else use all files in folder
        if file_list:
            self.files = file_list
        elif file_mapper:
            self.files = file_mapper(str(self.folder_name.resolve()))
        else:
            self.files = [str(f.resolve()) for f in self.folder_name.glob(f"*.{file_type}")]

        self.files = sorted(self.files)
        self.transform_x = transform_x
        self.transform_y = transform_y
        if label_mapper:
            self.labels = label_mapper(self.files)

        self.cached_data = {}
        print(len(self.files),len(self.labels))

    def __len__(self):
        return len(self.files)
    
    def _get_file_path(self,filename):
        return str(Path(self.folder_name,filename).resolve())
    
    def __getitem__(self,idx):
        filename = self.files[idx]
        if self.labels:
            label=self.labels[idx]
        if filename in self.cached_data:
            x_data = self.cached_data[filename]
        else:
            filepath = self._get_file_path(filename)
            # Assumes data has header row
            x_data = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=1,usecols=[2,3])
            self.cached_data[filename] = x_data
        #print(x_data,label)
        if self.transform_x:
            for tr in self.transform_x:
                x_data = tr(x_data)
        if self.transform_y:
            for tr in self.transform_y:
                label = tr(label)
        #print(f"This is the label:{label}")
        if self.labels:
            return x_data, label
        else:
            return x_data

class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass

class BaseGazeDataModule(LightningDataModule, ABC):
    @property 
    @abstractmethod
    def x_transforms(self) -> None:
        pass

    @property 
    @abstractmethod
    def y_transforms(self) -> None:
        pass

    @property 
    @abstractmethod
    def label_mapper(self) -> None:
        pass

    @property 
    @abstractmethod
    def splitter(self) -> None:
        pass

    @property 
    @abstractmethod
    def file_mapper(self) -> None:
        pass
class SequenceToLabelDataModule(BaseKFoldDataModule, BaseGazeDataModule):

    def __init__(self,
                data_dir: str,
                label_filepath: str,
                label_col: str,
                test_dir: Optional[str] = None,
                train_dataset: Optional[Dataset] = None,
                test_dataset: Optional[Dataset] = None,
                train_fold: Optional[Dataset] = None,
                val_fold: Optional[Dataset] = None,
                sequence_length: int = 500,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory: bool = True,
                drop_last: bool = True,            
                ):
        super().__init__()
        self.data_dir = data_dir
        self.label_col = label_col
        self.label_df = get_label_df(label_filepath)
        self.test_dir = test_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        '''
        Download and save data, do some preprocessing before transforms are applited
        '''
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", "predict", None):
            dataset = SequenceToLabelDataset(self.data_dir, file_mapper=self.file_mapper, label_mapper=self.label_mapper, transform_x=self.x_transforms, transform_y=self.y_transforms)
            if self.test_dir:
                self.train_dataset = dataset
            else:
                splits = train_test_split(np.arange(len(dataset.files)), dataset.labels, test_size=0.15, stratify=dataset.labels)
                self.train_dataset = Subset(dataset, splits[0])
                self.test_dataset = Subset(dataset, splits[1])
        elif stage == "test":
            if self.test_dir:
                self.test_dataset = SequenceToLabelDataset(dir, file_mapper=self.file_mapper, label_mapper=self.label_mapper, transform_x=self.x_transforms, transform_y=self.y_transforms)
            
            assert self.test_dataset is not None


    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset.dataset, train_indices)
        self.val_fold = Subset(self.train_dataset.dataset, val_indices)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        train_indices = self.train_dataset.indices
        train_files, train_labels = self.train_dataset.dataset.get_files_from_indices(train_indices), self.train_dataset.dataset.get_labels_from_indices(train_indices)
        if num_folds == 1:
            split_list = train_test_split(train_indices, test_size=0.15, stratify=train_labels)
            self.splits = [(split_list[i], split_list[i+1]) for i in range(0, len(split_list)-1)]
        elif num_folds > 1:
            splits = []
            for split in get_stratified_group_splits(train_files, self.label_df, self.label_col, folds=num_folds):
                train_split = [train_indices[ind] for ind in split[0]]
                val_split = [train_indices[ind] for ind in split[1]]
                splits.append((train_split, val_split))
            self.splits = splits
            
        else:
            raise ValueError
            
    def train_dataloader(self) -> DataLoader:
        if self.train_fold:
            return self._get_dataloader(self.train_fold)
        else:
            return self._get_dataloader(self.train_dataset)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._get_dataloader(self.val_fold)

    def predict_dataloader(self) -> DataLoader:
        if self.test_dataset:
            return self._get_dataloader(self.test_dataset)
        elif self.val_fold:
            return self._get_dataloader(self.val_fold)
        else:
            return self._get_dataloader(self.train_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset)

    def _get_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last, 
            pin_memory=self.pin_memory)

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SequenceToLabelDataModule")
        group.add_argument("--data_dir", type=str)
        group.add_argument("--test_dir", type=str, default="")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--label_filepath", type=str)
        group.add_argument("--label_col", type=str)
        group.add_argument("--sequence_length", type=int, default=500)
        return parent_parser

    @property
    def x_transforms(self):
        return T.Compose([LimitSequenceLength(self.sequence_length), ToTensor()])

    @property
    def y_transforms(self):
        return ToTensor()
    
    @property
    def label_mapper(self):
        return get_label_mapper(self.label_df, self.label_col)
    
    @property
    def splitter(self, num_folds, split_fn):
        return split_fn(num_folds)
    
    @property
    def file_mapper(self):
        return partial(get_filenames_for_dataset,label_df=self.label_df, label_col=self.label_col)

        
        
        

class GazeDataModule(LightningDataModule):
    def __init__(self, 
                data_dir: str,
                test_dir: str = None,
                transform_x:  Callable = None, 
                transform_y:  Callable = None,
                splitter: Callable = None,
                label_mapper: Callable = None,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory = True,
                drop_last = True,
                usecols: list = None,
                skiprows: int = 1,
                *args: Any,
                **kwargs: Any
                ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.splitter = splitter
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.batch_size = batch_size
        self.label_mapper = label_mapper
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.usecols = usecols
        self.skiprows = skiprows

    def prepare_data(self):
        '''
        Download and save data, do some preprocessing before transforms are applited
        '''
        pass

    def setup(self, stage: Optional[str] = None,):
        if stage in ("fit", None):
            dataset_full = SequenceToLabelDataset(self.data_dir, file_list=self.file_list, label_mapper=self.label_mapper, transform_x=self.transform_x, transform_y=self.transform_y, usecols=self.usecols, skiprows=self.skiprows)
            if self.splitter:
                self.dataset_train, self.dataset_val = self._subset_from_split(dataset_full)
            else:
                self.dataset_train = dataset_full

        if stage == "test":
            dir = self.test_dir if self.test_dir is not None else self.data_dir 
            self.dataset_test = SequenceToLabelDataset(dir, file_list=self.file_list, label_mapper=self.label_mapper, transform_x=self.transform_x, transform_y=self.transform_y, usecols=self.usecols, skiprows=self.skiprows)
    
    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.dataset_train)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._get_dataloader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        # If there is a test dataset then use that 
        if hasattr(self, "dataset_test"):
            return self._get_dataloader(self.dataset_test)
        # If there isn't then use the train dataset
        else:
            return self._get_dataloader(self.dataset_train)

    def _subset_from_split(self, ds):
        splits = self.splitter(ds)
        subsets = [Subset(ds, split) for split in splits]
        return subsets

    def _get_dataloader(self, dataset: Dataset):
            return DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                drop_last=self.drop_last, 
                pin_memory=self.pin_memory)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("GazeDataModule")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--test_dir", type=str, default="")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--usecols", type=int, nargs='*', default=None)
        parser.add_argument("--skiprows", type=int, default=1)
        return parser
