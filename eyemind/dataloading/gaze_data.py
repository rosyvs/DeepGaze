import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
from torch.utils.data import Dataset, DataLoader, Sampler
from pytorch_lightning import LightningDataModule

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



class GazeDataModule(LightningDataModule):
    def __init__(self, 
                data_dir: str,
                test_dir: str = None,
                transform_x: List[Callable] = None, 
                transform_y: List[Callable] = None,
                file_list: List[str] = [],
                label_mapper: Callable = None,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory = True,
                drop_last = True,
                *args: Any,
                **kwargs: Any
                ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.file_list = file_list
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.batch_size = batch_size
        self.label_mapper = label_mapper
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        '''
        Download and save data, do some preprocessing before transforms are applited
        '''
        pass

    def setup(self, stage: Optional[str] = None,):
        if stage in ("fit", None):
            self.dataset_train = MultiFileDataset(self.data_dir, file_list=self.file_list, label_mapper=self.label_mapper, transform_x=self.transform_x, transform_y=self.transform_y)
        if stage == "test":
            dir = self.test_dir if self.test_dir is not None else self.data_dir 
            self.dataset_test = MultiFileDataset(dir, file_list=self.file_list, label_mapper=self.label_mapper, transform_x=self.transform_x, transform_y=self.transform_y)
    
    def train_dataloader(self, shuffle: bool = False, sampler: Sampler=None) -> DataLoader:
        return self._get_dataloader(self.dataset_train, sampler=sampler, shuffle=shuffle)

    def val_dataloader(self, shuffle: bool = False, sampler: Sampler=None) -> Union[DataLoader, List[DataLoader]]:
        # If there is a validation dataset then use that 
        if hasattr(self, "dataset_val"):
            return self._get_dataloader(self.dataset_val, sampler=sampler, shuffle=shuffle)
        # If there isn't then use the train dataset: this is used when doing crossvalidation with sampler
        else:
            return self._get_dataloader(self.dataset_train, sampler=sampler, shuffle=shuffle)

    def test_dataloader(self, shuffle: bool = False, sampler: Sampler=None) -> DataLoader:
        # If there is a test dataset then use that 
        if hasattr(self, "dataset_test"):
            return self._get_dataloader(self.dataset_test, sampler, shuffle)
        # If there isn't then use the train dataset
        else:
            return self._get_dataloader(self.dataset_train, sampler=sampler, shuffle=shuffle)

    def _get_dataloader(self, dataset: Dataset, sampler: Sampler=None, shuffle: bool = False):
        if sampler:
            return DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                drop_last=self.drop_last, 
                pin_memory=self.pin_memory,
                sampler=sampler)
        else:
            return DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                drop_last=self.drop_last, 
                pin_memory=self.pin_memory,
                shuffle=shuffle)