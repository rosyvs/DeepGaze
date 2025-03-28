
from abc import ABC, abstractmethod
from functools import partial
import os
import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from eyemind.dataloading.transforms import StandardScaler, GazeScaler
from eyemind.dataloading.load_dataset import label_samples, filter_files_by_seqlen, get_filenames_for_dataset, get_label_df, get_label_mapper, get_participant_splits, get_stratified_group_splits, limit_sequence_len, load_file_folds, write_splits, label_samples_and_files
from eyemind.dataloading.batch_loading import seq2seq_collate_fn, multitask_collate_fn, split_collate_fn, multilabel_multitask_collate_fn, variable_length_seq2seq_collate_fn, variable_length_multilabel_multitask_collate_fn, variable_length_multitask_collate_fn, variable_length_seq2label_collate_fn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler, Subset

from pytorch_lightning import LightningDataModule

from eyemind.dataloading.transforms import LimitSequenceLength, GazeScaler, ToTensor


class SequenceLabelDataset(Dataset):
    def __init__(self, 
            folder_name, 
            file_list=[], 
            file_mapper=None, 
            file_type="csv", 
            transform_x=None, 
            transform_y=None, 
            label_mapper=None, 
            skiprows=1, 
            usecols=[1,2], #?? TODO: cols 1 and 2 are XAvg and YAvg ONLY IF NO INDEX COL IN CSV
            scale_gaze=False,
            gaze_scaler=None, 
):
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
        self.scale_gaze=scale_gaze
        self.gaze_scaler=gaze_scaler
        # If there is a list passed then use it, else if function then use it, else use all files in folder
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
            self.labels = label_mapper(filenames=self.files)
        else:
            self.labels = None
        
    def __len__(self):
        return len(self.files)
    
    def _get_file_path(self,filename):
        return str(Path(self.folder_name,filename).resolve())
    
    def __getitem__(self,idx):
        filename = self.files[idx]
        filepath = self._get_file_path(filename)
        x_data = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=self.skiprows,usecols=self.usecols)
        if self.scale_gaze:
            x_data = self.gaze_scaler(x_data)
        if self.transform_x:
            x_data = self.transform_x(x_data)
        if self.labels:
            label=self.labels[idx]
            if self.transform_y:
                label = self.transform_y(label)
            return x_data, label
        else:
            return x_data, None

    def get_labels_from_indices(self, indices):
        return [self.labels[ind] for ind in indices]
    
    def get_files_from_indices(self, indices):
        return [self.files[ind] for ind in indices]

    def get_indices_from_files(self, files):
        file_index_map = {f: i for i, f in enumerate(self.files)}
        return [file_index_map[f] for f in files if f in file_index_map]

    @classmethod
    def load_dataset(cls, path, **kwargs):
        pass

class SequenceMultiLabelDataset(SequenceLabelDataset):

    '''
    Dataset for large data with multiple csv files AND multile labels per sequence 
    (curently supports 1 sequence(file)-level and 1 sample-level label)

    Args:
    folder_name (str): path to folder where csv files are for x data
    file_list (list[str]): list of filenames to use for dataset
    file_mapper (fn): Gives list of files in folder to use for the dataset. Returns filename as string
    file_type (str): file extension. This is only used if file_mapper is none
    transform_x (list[fns]): functions that are applied to the x_data
    label_file (str): path to file specifying labels for each file
    label_mapper (fn): maps list of files to labels. Returns list
    '''
    def __init__(self, 
                folder_name, 
                file_list=[], 
                file_mapper=None, 
                file_type="csv", 
                transform_x=None, 
                transform_y=None, 
                label_mapper=None, 
                skiprows=1, 
                usecols=[1,2], #?? TODO: cols 1 and 2 are XAvg and YAvg ONLY IF NO INDEX COL IN CSV
                gaze_scaler=None, 
                file_label_scaler=None, 
                sample_label_scaler=None, 
                ):
        self.folder_name = Path(folder_name)
        self.skiprows = skiprows
        self.usecols = usecols
        # If there is a list passed then use it, else if function then use it, else use all files in folder
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
            self.labels = label_mapper(filenames=self.files)
        self.gaze_scaler=gaze_scaler
        self.sample_label_scaler=sample_label_scaler
        self.file_label_scaler=file_label_scaler

    def __getitem__(self,idx):
        filename = self.files[idx]
        filepath = self._get_file_path(filename)
        x_data = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=self.skiprows,usecols=self.usecols)
        if self.gaze_scaler:
            x_data = self.gaze_scaler(x_data)
        if self.transform_x:
            x_data = self.transform_x(x_data)
        if self.labels:
            sample_label, file_label =self.labels[idx]
            if self.file_label_scaler:
                file_label=self.file_label_scaler(file_label)
            if self.sample_label_scaler:
                sample_label=self.sample_label_scaler(sample_label)
            if self.transform_y:
                sample_label = self.transform_y(sample_label)
                file_label = self.transform_y(file_label)
            label = (sample_label, file_label)
            return x_data, label
        else:
            return x_data

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
            self.labels = label_mapper(files=self.files)

        self.cached_data = {}
        print(len(self.files),len(self.labels))
    
    def export_selected_data(self):
        self.files

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
            x_data = np.loadtxt(open(filepath,"rb"),delimiter=",",skiprows=1,usecols=[1,2])
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

    def save_folds(self, path):
        file_folds = [(self.train_dataset.dataset.get_files_from_indices(split[0]), 
            self.train_dataset.dataset.get_files_from_indices(split[1])) for split in self.splits]
        write_splits(file_folds, path, folds=True)

    def load_folds(self, path):
        file_folds = load_file_folds(path)
        self.splits = [(self.train_dataset.dataset.get_indices_from_files(fold[0]), self.train_dataset.dataset.get_indices_from_files(fold[1])) for fold in file_folds]
        
class GroupStratifiedKFoldDataModule(BaseKFoldDataModule):
    
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

class ParticipantKFoldDataModule(BaseKFoldDataModule):
    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset.dataset, train_indices)
        self.val_fold = Subset(self.train_dataset.dataset, val_indices)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        train_indices = self.train_dataset.indices
        train_files, train_labels = self.train_dataset.dataset.get_files_from_indices(train_indices), self.train_dataset.dataset.get_labels_from_indices(train_indices)
        if num_folds == 1:
            split_list = train_test_split(train_indices, test_size=0.15)
            self.splits = [(split_list[i], split_list[i+1]) for i in range(0, len(split_list)-1)]
        elif num_folds > 1:
            splits = []
            for split in get_participant_splits(train_files, self.label_df, folds=num_folds):
                train_split = [train_indices[ind] for ind in split[0]]
                val_split = [train_indices[ind] for ind in split[1]]
                splits.append((train_split, val_split))
            self.splits = splits
            
        else:
            raise ValueError   
            
class GroupStratifiedNestedCVDataModule(GroupStratifiedKFoldDataModule):

    def setup_cv_folds(self, num_outer_folds: int, num_inner_folds: int) -> None:
        train_indices = self.train_dataset.indices
        train_files, train_labels = self.train_dataset.dataset.get_files_from_indices(train_indices), self.train_dataset.dataset.get_labels_from_indices(train_indices)        
        if num_outer_folds == 1:
            self.outer_splits = train_test_split(train_indices, test_size=0.15, stratify=train_labels)
        elif num_outer_folds > 1:
            outer_splits = []
            inner_splits_list = []
            for outer_split in get_stratified_group_splits(train_files, self.label_df, self.label_col, folds=num_outer_folds):
                train_split = [train_indices[ind] for ind in outer_split[0]]
                val_split = [train_indices[ind] for ind in outer_split[1]]
                outer_splits.append((train_split, val_split))
                inner_splits = []
                outer_split_train_files = self.train_dataset.dataset.get_files_from_indices(train_split)
                for inner_split in get_stratified_group_splits(outer_split_train_files, self.label_df, self.label_col, folds=num_inner_folds):
                    train_split = [train_indices[ind] for ind in inner_split[0]]
                    val_split = [train_indices[ind] for ind in inner_split[1]]
                    inner_splits.append((train_split, val_split))
                inner_splits_list.append(inner_splits)
            self.outer_splits = outer_splits
            self.inner_splits = inner_splits_list
    
    def setup_cv_fold_index(self, outer_fold_index: int, inner_fold_index: int) -> None:
        # Use inner fold
        if inner_fold_index >= 0:
            train_indices, val_indices = self.inner_splits[outer_fold_index][inner_fold_index]
        # Use outer fold
        else:
            train_indices, val_indices = self.outer_splits[outer_fold_index]
        self.train_fold = Subset(self.train_dataset.dataset, train_indices)
        self.val_fold = Subset(self.train_dataset.dataset, val_indices)

    def get_cv_fold(self, outer_fold_index: int, inner_fold_index: int):
        if inner_fold_index >= 0:
            train_indices, val_indices = self.inner_splits[outer_fold_index][inner_fold_index]
        # Use outer fold
        else:
            train_indices, val_indices = self.outer_splits[outer_fold_index]
        return Subset(self.train_dataset.dataset, train_indices), Subset(self.train_dataset.dataset, val_indices)


class BaseGazeDataModule(LightningDataModule, ABC):
    
    def save_setup(self, path):
        train_files = None
        val_files = None
        test_files = None
        if isinstance (self.train_dataset, Subset):
            train_files = self.train_dataset.dataset.get_files_from_indices(self.train_dataset.indices)
        else:
            train_files = self.train_dataset.files
        if hasattr(self, "val_dataset"):
            if isinstance (self.val_dataset, Subset):
                val_files = self.val_dataset.dataset.get_files_from_indices(self.val_dataset.indices)
            else:
                val_files = self.val_dataset.files
        if hasattr(self, "test_dataset"):        
            if isinstance (self.test_dataset, Subset):
                test_files = self.test_dataset.dataset.get_files_from_indices(self.test_dataset.indices)
            else:
                test_files = self.test_dataset.files              
        save_dict = {dataset_str: files_list for dataset_str,files_list in zip(["train", "val", "test"],[train_files, val_files, test_files]) if files_list}
        with open(path, 'w') as f:
            yaml.safe_dump(save_dict, f)        

    def load_setup(self, dataset):
        with open(self.load_setup_path, 'r') as f:
            file_dict = yaml.safe_load(f)              
        for k,v in file_dict.items():
            split = dataset.get_indices_from_files(v)
            if k == "train":
                self.train_dataset = Subset(dataset, split)
            elif k == "val":
                self.val_dataset = Subset(dataset, split)
            elif k == "test":
                self.test_dataset = Subset(dataset, split)
            else:
                raise ValueError("File doesn't have train, val, or test files")
    
    def get_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last,
            persistent_workers=True, 
            pin_memory=self.pin_memory)

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
    def file_mapper(self) -> None:
        pass
class SequenceToLabelDataModule(GroupStratifiedNestedCVDataModule, BaseGazeDataModule):
    # Note: this doesnt have a special collate_fn but it does limit sequence length in x.transforms by takng first sequence-Length elements
    def __init__(self,
                data_dir: str,
                label_filepath: str,
                label_col: str,
                load_setup_path: Optional[str] = None,
                test_dir: Optional[str] = None,
                train_dataset: Optional[Dataset] = None,
                test_dataset: Optional[Dataset] = None,
                val_dataset: Optional[Dataset] = None,
                train_fold: Optional[Dataset] = None,
                val_fold: Optional[Dataset] = None,
                sequence_length: int = 500,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory: bool = True,
                drop_last: bool = True,
                scale_gaze: Optional[bool] = False,
                mean_gaze_xy: Optional[list]=[-0.698, -1.940],
                std_gaze_xy: Optional[list]=[4.15, 3.286],
                usecols: list = [1,2]          
                ):
        super().__init__()
        self.data_dir = data_dir
        self.label_col = label_col
        self.label_df = get_label_df(label_filepath)
        self.load_setup_path = load_setup_path
        self.test_dir = test_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.scale_gaze =scale_gaze
        self.mean_gaze_xy=mean_gaze_xy
        self.std_gaze_xy=std_gaze_xy        
        self.usecols=usecols

    def prepare_data(self):
        '''
        Download and save data, do some preprocessing before transforms are applited
        '''
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", "predict", None):
            dataset = SequenceLabelDataset(
                self.data_dir, 
                file_mapper=self.file_mapper, 
                label_mapper=self.label_mapper, 
                transform_x=self.x_transforms, 
                transform_y=self.y_transforms, 
                gaze_scaler=self.gaze_scaler if self.scale_gaze else None,
                usecols=self.usecols)
            if self.load_setup_path:
                self.load_setup(dataset)
            else:
                if self.test_dir:
                    self.train_dataset = dataset
                else:
                    train_val_splits, test_split = train_test_split(np.arange(len(dataset.files)), test_size=0.1)
                    train_split, val_split = train_test_split(train_val_splits, test_size=0.2)
                    self.splits = (train_split, val_split, test_split)
                    self.test_dataset = Subset(dataset, self.splits[2])
                self.train_dataset = Subset(dataset, self.splits[0])
                self.val_fold = Subset(dataset, self.splits[1])
  
        elif stage == "test":
            if self.test_dir:
                self.test_dataset = SequenceLabelDataset(
                dir, 
                file_mapper=self.file_mapper, 
                label_mapper=self.label_mapper, 
                transform_x=self.x_transforms, 
                transform_y=self.y_transforms,
                scale_gaze=self.scale_gaze,
                gaze_scaler=self.gaze_scaler if self.scale_gaze else None,
                usecols=self.usecols)
            
            assert self.test_dataset is not None

    def train_dataloader(self) -> DataLoader:
        if self.train_fold:
            return self.get_dataloader(self.train_fold)
        else:
            return self.get_dataloader(self.train_dataset)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.get_dataloader(self.val_fold)

    def predict_dataloader(self) -> DataLoader:
        if self.test_dataset:
            return self.get_dataloader(self.test_dataset)
        elif self.val_fold:
            return self.get_dataloader(self.val_fold)
        else:
            return self.get_dataloader(self.train_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset)

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
        group.add_argument("--usecols", type=list, default=[1,2])
        group.add_argument("--scale_gaze", type=bool, default=False)
        group.add_argument("--mean_gaze_xy",nargs='*', default=None)
        group.add_argument("--std_gaze_xy",nargs='*', default=None)
        return parent_parser

    @property
    def x_transforms(self):
        return T.Compose([LimitSequenceLength(self.sequence_length), ToTensor()]) 

    @property
    def y_transforms(self):
        return ToTensor()
    
    @property
    def label_mapper(self):
        if self.label_col:
            return get_label_mapper(self.label_df, self.label_col)
        else:
            return None
    
    @property
    def file_mapper(self):
        return partial(get_filenames_for_dataset,label_df=self.label_df, label_col=self.label_col)

    @property
    def gaze_scaler(self):
        if self.scale_gaze:
            scaler=partial(GazeScaler(mean=self.mean_gaze_xy, std=self.std_gaze_xy))
        else:
            scaler=None
        return scaler

        
class BaseSequenceToSequenceDataModule(BaseGazeDataModule):

    def __init__(self,
                data_dir: str,
                label_filepath: str,
                sample_label_col: str,
                load_setup_path: Optional[str] = None,
                test_dir: Optional[str] = None,
                train_dataset: Optional[Dataset] = None,
                val_dataset: Optional[Dataset] = None,
                test_dataset: Optional[Dataset] = None,
                sequence_length: int = 500,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory: bool = True,
                drop_last: bool = True,
                contrastive: bool = False,
                min_sequence_length: int = 500,            
                ):
        super().__init__()
        self.data_dir = data_dir
        self.label_df = get_label_df(label_filepath)
        self.sample_label_col = sample_label_col #TODO: needed?
        self.load_setup_path = load_setup_path
        self.test_dir = test_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.contrastive = contrastive
        self.min_sequence_length = min_sequence_length

    def prepare_data(self):
        '''
        Download and save data, do some preprocessing before transforms are applited
        '''
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", "predict", None):
            dataset = SequenceLabelDataset(
            self.data_dir, 
            file_mapper=self.file_mapper, 
            label_mapper=self.label_mapper, 
            transform_x=self.x_transforms, 
            transform_y=self.y_transforms, 
            usecols=[1,2]
            ) #TODO: these are hardcoded, but the default is 1,2. 2,3 is getting YAvg and event for my files with no idenx col, but ricks had index col
            if self.load_setup_path:
                self.load_setup(dataset)
            else:
                if self.test_dir:
                    self.splits = train_test_split(np.arange(len(dataset.files)), test_size=0.2)
                else:
                    train_val_splits, test_split = train_test_split(np.arange(len(dataset.files)), test_size=0.1)
                    train_split, val_split = train_test_split(train_val_splits, test_size=0.2)
                    self.splits = (train_split, val_split, test_split)
                    self.test_dataset = Subset(dataset, self.splits[2])
                self.train_dataset = Subset(dataset, self.splits[0])
                self.val_dataset = Subset(dataset, self.splits[1])

        elif stage == "test":
            if self.test_dir:
                self.test_dataset = SequenceLabelDataset(dir, file_mapper=self.file_mapper, label_mapper=self.label_mapper, transform_x=self.x_transforms, transform_y=self.y_transforms)
            assert self.test_dataset is not None
            
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.get_dataloader(self.val_dataset)

    def predict_dataloader(self) -> DataLoader:
        if self.test_dataset:
            return self.get_dataloader(self.test_dataset)
        elif self.val_dataset:
            return self.get_dataloader(self.val_dataset)
        else:
            return self.get_dataloader(self.train_dataset) 

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset)

    def get_dataloader(self, dataset: Dataset):
        if self.contrastive:
            collate_fn = multitask_collate_fn
        else:
            collate_fn = seq2seq_collate_fn
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last, 
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, self.sequence_length))

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        group = parent_parser.add_argument_group("BaseSequenceToSequenceDataModule")
        group.add_argument("--data_dir", type=str)
        group.add_argument("--test_dir", type=str, default="")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--label_filepath", type=str)
        group.add_argument("--sample_label_col", type=str)
        group.add_argument("--sequence_length", type=int, default=500)
        group.add_argument("--contrastive", type=bool, default=False)
        group.add_argument("--min_sequence_length", type=int, default=500)
        return parent_parser

    @property
    def x_transforms(self):
        return ToTensor()

    @property
    def y_transforms(self):
        return ToTensor()

    @property
    def label_mapper(self):
        return partial(label_samples, folder=self.data_dir, label_col=self.sample_label_col)
    
    @property
    def file_mapper(self):
        return partial(filter_files_by_seqlen, self.label_df, min_sequence_length=self.min_sequence_length)

class SequenceToSequenceDataModule(GroupStratifiedKFoldDataModule, BaseGazeDataModule):

    def __init__(self,
                data_dir: str,
                label_filepath: str,
                sample_label_col: str,
                load_setup_path: Optional[str] = None,
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
        super().__init__(data_dir=data_dir, label_filepath=label_filepath,label_col=sample_label_col)
        self.data_dir = data_dir
        self.label_filepath=label_filepath
        self.label_df = get_label_df(label_filepath)
        self.sample_label_col=sample_label_col
        self.load_setup_path = load_setup_path
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
            dataset = SequenceLabelDataset(
                self.data_dir, 
                file_mapper=self.file_mapper, 
                label_mapper=self.label_mapper, 
                transform_x=self.x_transforms, 
                transform_y=self.y_transforms)
            if self.load_setup_path:
                self.load_setup(dataset)
            else:
                if self.test_dir:
                    self.train_dataset = dataset
                else:
                    splits = train_test_split(np.arange(len(dataset.files)), dataset.labels, test_size=0.15)
                    self.train_dataset = Subset(dataset, splits[0])
                    self.test_dataset = Subset(dataset, splits[1])
        elif stage == "test":
            if self.test_dir:
                self.test_dataset = SequenceLabelDataset(
                    dir, 
                    file_mapper=self.file_mapper, 
                    label_mapper=self.label_mapper, 
                    transform_x=self.x_transforms, 
                    transform_y=self.y_transforms)
            assert self.test_dataset is not None
            
    def train_dataloader(self) -> DataLoader:
        if self.train_fold:
            return self.get_dataloader(self.train_fold)
        else:
            return self.get_dataloader(self.train_dataset)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.get_dataloader(self.val_fold)

    def predict_dataloader(self) -> DataLoader:
        if self.test_dataset:
            return self.get_dataloader(self.test_dataset)
        elif self.val_fold:
            return self.get_dataloader(self.val_fold)
        else:
            return self.get_dataloader(self.train_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset)

    def get_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last, 
            pin_memory=self.pin_memory,
            collate_fn=partial(split_collate_fn, self.sequence_length))

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SequenceToSequenceDataModule")
        group.add_argument("--data_dir", type=str)
        group.add_argument("--test_dir", type=str, default="")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--label_filepath", type=str)
        group.add_argument("--sample_label_col", type=str)
        group.add_argument("--sequence_length", type=int, default=500)
        return parent_parser

    @property
    def x_transforms(self):
        return ToTensor()

    @property
    def y_transforms(self):
        return ToTensor()

    @property
    def label_mapper(self):
        return partial(label_samples, folder=self.data_dir, label_col=self.sample_label_col)
    
    @property
    def file_mapper(self):
        return partial(filter_files_by_seqlen, self.label_df)

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
            dataset_full = SequenceLabelDataset(
                self.data_dir, 
                file_list=self.file_list, 
                label_mapper=self.label_mapper, 
                transform_x=self.transform_x, 
                transform_y=self.transform_y, 
                usecols=self.usecols, 
                skiprows=self.skiprows)
            if self.splitter:
                self.dataset_train, self.dataset_val = self._subset_from_split(dataset_full)
            else:
                self.dataset_train = dataset_full

        if stage == "test":
            dir = self.test_dir if self.test_dir is not None else self.data_dir 
            self.dataset_test = SequenceLabelDataset(
                dir, 
                file_list=self.file_list, 
                label_mapper=self.label_mapper, 
                transform_x=self.transform_x, 
                transform_y=self.transform_y, 
                usecols=self.usecols, 
                skiprows=self.skiprows)
    
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

class PIDkFoldS2SDataModule(BaseSequenceToSequenceDataModule, ParticipantKFoldDataModule):
    pass

class SequenceToMultiLabelDataModule(SequenceToSequenceDataModule, SequenceToLabelDataModule):
    # supporting both seq2seq label and seq2label
    def __init__(self,
                data_dir: str,
                label_filepath: str,
                sample_label_col: Optional[str] = None,
                file_label_col: Optional[str] = None,
                load_setup_path: Optional[str] = None,
                test_dir: Optional[str] = None,
                train_dataset: Optional[Dataset] = None,
                val_dataset: Optional[Dataset] = None,
                test_dataset: Optional[Dataset] = None,
                train_fold: Optional[Dataset] = None,
                val_fold: Optional[Dataset] = None,
                sequence_length: int = 500,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory: bool = True,
                drop_last: bool = True,   
                label_length: int = 48,
                pred_length: Optional[int] = None,    
                min_sequence_length: int = 500,
                contrastive: bool = True,    
                scale_file_label: Optional[bool] = True,
                scale_sample_label: Optional[bool] = False,
                scale_gaze: Optional[bool] = False,
                mean_gaze_xy: Optional[list]=[-0.698, -1.940],
                std_gaze_xy: Optional[list]=[4.15, 3.286],
                mean_sample_label: Optional[float]=0.0,
                std_sample_label: Optional[float]=1.0,
                usecols: Optional[list]=[1,2],

                ):
        super().__init__(data_dir=data_dir, 
                        label_filepath=label_filepath, 
                        sample_label_col=sample_label_col)
        # self.data_dir = data_dir
        self.label_df = get_label_df(label_filepath)
        # self.sample_label_col=sample_label_col
        self.file_label_col=file_label_col
        self.load_setup_path = load_setup_path
        self.test_dir = test_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.pred_length = pred_length
        self.label_length = label_length
        self.contrastive = contrastive
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.scale_file_label = scale_file_label
        self.scale_sample_label =scale_sample_label
        self.scale_gaze =scale_gaze
        self.mean_gaze_xy=mean_gaze_xy
        self.std_gaze_xy=std_gaze_xy
        self.mean_sample_label=mean_sample_label
        self.std_sample_label=std_sample_label
        self.usecols=usecols

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", "predict", None):
            if self.file_label_col:
                dataset = SequenceMultiLabelDataset(
                    self.data_dir, 
                    file_mapper=self.file_mapper, 
                    label_mapper=self.label_mapper, 
                    transform_x=self.x_transforms, 
                    transform_y=self.y_transforms,
                    gaze_scaler=self.gaze_scaler if self.scale_gaze else None,
                    file_label_scaler=self.file_label_scaler if self.scale_file_label else None,
                    sample_label_scaler=self.sample_label_scaler if self.scale_sample_label else None,
                    usecols=self.usecols)
            else:
                dataset = SequenceLabelDataset(
                self.data_dir, 
                    file_mapper=self.file_mapper, 
                    label_mapper=self.label_mapper, 
                    transform_x=self.x_transforms, 
                    transform_y=self.y_transforms, 
                    scale_gaze=self.scale_gaze,
                    gaze_scaler=self.gaze_scaler if self.scale_gaze else None,
                    usecols=self.usecols
                    )
            if self.load_setup_path:
                self.load_setup(dataset)
            else:
                if self.test_dir:
                    self.splits = train_test_split(np.arange(len(dataset.files)), test_size=0.2)
                else:
                    train_val_splits, test_split = train_test_split(np.arange(len(dataset.files)), test_size=0.1)
                    train_split, val_split = train_test_split(train_val_splits, test_size=0.2)
                    self.splits = (train_split, val_split, test_split)
                    self.test_dataset = Subset(dataset, self.splits[2])
                self.train_dataset = Subset(dataset, self.splits[0])
                self.val_dataset = Subset(dataset, self.splits[1])

        # elif stage == "test":
        #     if self.test_dir:
        #         self.test_dataset = SequenceLabelDataset(
        #             dir, 
        #             file_mapper=self.file_mapper, 
        #             label_mapper=self.label_mapper, 
        #             transform_x=self.x_transforms, 
        #             transform_y=self.y_transforms)
        #     assert self.test_dataset is not None

    def train_dataloader(self) -> DataLoader:
        if self.train_fold:
            return self.get_dataloader(self.train_fold)
        else:
            return self.get_dataloader(self.train_dataset)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.val_fold:
            return self.get_dataloader(self.val_fold)
        else:
            return self.get_dataloader(self.val_dataset)

    def predict_dataloader(self) -> DataLoader:
        if self.test_dataset:
            return self.get_dataloader(self.test_dataset)
        elif self.val_fold:
            return self.get_dataloader(self.val_fold)
        else:
            return self.get_dataloader(self.train_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset)

    def get_dataloader(self, dataset: Dataset):
        if self.file_label_col:
            collate_fn=partial(multilabel_multitask_collate_fn, self.sequence_length)
        else:
            collate_fn=partial(multitask_collate_fn, self.sequence_length)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last, 
            pin_memory=self.pin_memory,
            collate_fn=collate_fn)

    def setup_fold_index(self, fold_index: int) -> None:# From ParticipantKfold data module to override group stratified k fold's method
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset.dataset, train_indices)
        self.val_fold = Subset(self.train_dataset.dataset, val_indices)

    def setup_folds(self, num_folds: int) -> None: # From ParticipantKfold data module to override group stratified k fold's method
        self.num_folds = num_folds
        train_indices = self.train_dataset.indices
        train_files, train_labels = self.train_dataset.dataset.get_files_from_indices(train_indices), self.train_dataset.dataset.get_labels_from_indices(train_indices)
        if num_folds == 1:
            split_list = train_test_split(train_indices, test_size=0.15)
            self.splits = [(split_list[i], split_list[i+1]) for i in range(0, len(split_list)-1)]
        elif num_folds > 1:
            splits = []
            for split in get_participant_splits(train_files, self.label_df, folds=num_folds):
                train_split = [train_indices[ind] for ind in split[0]]
                val_split = [train_indices[ind] for ind in split[1]]
                splits.append((train_split, val_split))
            self.splits = splits
            
        else:
            raise ValueError   

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SequenceToMultiLabelDataModule")
        group.add_argument("--data_dir", type=str)
        group.add_argument("--test_dir", type=str, default="")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--label_filepath", type=str)
        group.add_argument("--sample_label_col", type=str)
        group.add_argument("--file_label_col", type=str, default=None)
        group.add_argument("--sequence_length", type=int, default=500)
        group.add_argument("--min_sequence_length", type=int, default=500)
        group.add_argument("--contrastive", type=bool, default=False)
        group.add_argument("--scale_file_label", type=bool, default=False)
        group.add_argument("--scale_sample_label", type=bool, default=False)
        group.add_argument("--scale_gaze", type=bool, default=False)
        group.add_argument("--mean_gaze_xy",nargs='*', default=None)
        group.add_argument("--std_gaze_xy",nargs='*', default=None)
        group.add_argument("--mean_sample_label", type=float, default=0.0)
        group.add_argument("--std_sample_label", type=float, default=1.0)
        group.add_argument("--usecols", type=list, default=[1,2])
        return parent_parser
    @property
    def x_transforms(self):
        return ToTensor()
    @property
    def y_transforms(self):
        return ToTensor()
    @property
    def file_mapper(self):
        return partial(filter_files_by_seqlen, self.label_df, min_sequence_length=self.min_sequence_length)
    @property
    def label_mapper(self):
        if self.file_label_col:
            mapper = partial(label_samples_and_files, label_df=self.label_df, folder=self.data_dir, sample_label_col=self.sample_label_col, file_label_col=self.file_label_col)
        else:
            mapper = partial(label_samples, folder=self.data_dir, label_col=self.sample_label_col)
        return mapper
    @property
    def file_label_scaler(self):
        if self.scale_file_label:
            mn = np.mean(self.label_df[self.file_label_col])
            sd = np.std(self.label_df[self.file_label_col])
            scaler=partial(StandardScaler(mean=mn, std=sd))
        else:
            scaler=None
        return scaler
    @property
    def sample_label_scaler(self):
        if self.scale_sample_label:
            scaler=partial(StandardScaler(mean=self.mean_sample_label, std=std_sample_label))
        else:
            scaler=None
        return scaler
    @property
    def gaze_scaler(self):
        if self.scale_gaze:
            scaler=partial(GazeScaler(mean=self.mean_gaze_xy, std=self.std_gaze_xy))
        else:
            scaler=None 
        return scaler

class VariableLengthSequenceToSequenceDataModule(SequenceToSequenceDataModule):
    # initialise supercass
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # remove sequence_length
        self.sequence_length = None # because it is determined by min and max sequence length now
        # add min and max sequence length
        self.min_sequence_length = kwargs.get("min_sequence_length", 500)
        self.max_sequence_length = kwargs.get("max_sequence_length", 500)

    def get_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last, 
            pin_memory=self.pin_memory,
            collate_fn=partial(variable_length_seq2seq_collate_fn, self.max_sequence_length))

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        group = parent_parser.add_argument_group("VariableLengthSequenceToSequenceDataModule")
        group.add_argument("--min_sequence_length", type=int, default=500)
        group.add_argument("--max_sequence_length", type=int, default=500)
        return parent_parser


class VariableLengthSequenceToLabelDataModule(SequenceToLabelDataModule):
    # initialise supercass
    def __init__(self, *, min_sequence_length=500, max_sequence_length=500, **kwargs):
        super().__init__( **kwargs)
        # remove sequence_length
        self.sequence_length = None # because it is determined by min and max sequence length now
        # add min and max sequence length
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

    def get_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last, 
            pin_memory=self.pin_memory,
            collate_fn=partial(variable_length_seq2label_collate_fn, self.max_sequence_length))
    
    @property # overridign the x_transform with limit sequence length
    def x_transforms(self):
        return ToTensor()

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        group = parent_parser.add_argument_group("VariableLengthSequenceToLabelDataModule")
        group.add_argument("--min_sequence_length", type=int, default=500)
        group.add_argument("--max_sequence_length", type=int, default=500)
        return parent_parser

#TODO: VariableLengthSequenceToMultiLabelDataModule