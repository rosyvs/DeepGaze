from functools import partial
import random
from typing import List, Optional, Union
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from eyemind.dataloading.gaze_data import BaseSequenceToSequenceDataModule, ParticipantKFoldDataModule, SequenceLabelDataset, SequenceToMultiLabelDataModule, VariableLengthSequenceToLabelDataModule, VariableLengthSequenceToSequenceDataModule
from torch.utils.data import Dataset, DataLoader, Subset

from eyemind.dataloading.load_dataset import filter_files_by_seqlen, get_label_df, label_samples, label_files
from eyemind.dataloading.batch_loading import *
from eyemind.dataloading.transforms import ToTensor


class InformerDataModule(BaseSequenceToSequenceDataModule, ParticipantKFoldDataModule):

    def __init__(self,
                data_dir: str,
                label_filepath: str,
                file_label_col: str, # file-level label col in label_filepath 
                sample_label_col: str, # sample-level label col in each gaze file
                load_setup_path: Optional[str] = None,
                test_dir: Optional[str] = None,
                train_dataset: Optional[Dataset] = None,
                val_dataset: Optional[Dataset] = None,
                test_dataset: Optional[Dataset] = None,
                train_fold: Optional[Dataset] = None,
                val_fold: Optional[Dataset] = None,
                sequence_length: int = 500,
                label_length: int = 48,
                pred_length: Optional[int] = None,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory: bool = True,
                drop_last: bool = True,
                min_sequence_length: int = 500,
                contrastive: bool = False,
                scale_gaze: Optional[bool] = False,
                mean_gaze_xy: Optional[list]=[-0.698, -1.940],
                std_gaze_xy: Optional[list]=[4.15, 3.286],
                ):
        super().__init__(data_dir,
                        label_filepath,
                        sample_label_col,
                        load_setup_path,
                        test_dir,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        sequence_length,
                        num_workers,
                        batch_size,
                        pin_memory,
                        drop_last,
                        False,
                        min_sequence_length,
                        )
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.pred_length = pred_length
        self.label_length = label_length
        self.contrastive = contrastive
        self.sample_label_col=sample_label_col #TODO: needed??
        self.file_label_col=file_label_col #TODO: needed??
        self.scale_gaze =scale_gaze
        self.mean_gaze_xy=mean_gaze_xy
        self.std_gaze_xy=std_gaze_xy

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
                usecols=[1,2], 
                scale_gaze=self.scale_gaze,
                gaze_scaler=self.gaze_scaler) # this is scaling X
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
        group = parent_parser.add_argument_group("InformerDataModule")
        group.add_argument("--scale_gaze", type=bool, default=False)
        group.add_argument("--mean_gaze_xy",nargs='*', default=None)
        group.add_argument("--std_gaze_xy",nargs='*', default=None)
        group.add_argument("--data_dir", type=str)
        group.add_argument("--test_dir", type=str, default="")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--label_filepath", type=str)
        group.add_argument("--sample_label_col", type=str)
        group.add_argument("--file_label_col", type=str)
        group.add_argument("--sequence_length", type=int, default=500)
        group.add_argument("--min_sequence_length", type=int, default=500)
        group.add_argument("--contrastive", type=bool, default=False)
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

    @property
    def gaze_scaler(self):
        if self.scale_gaze:
            scaler=partial(GazeScaler(mean=self.mean_gaze_xy, std=self.std_gaze_xy))
        else:
            scaler=None
        return scaler


# these classes can be the same as in gaze_data.py, no informer-specific changes needed
class InformerMultiLabelDatamodule(SequenceToMultiLabelDataModule):
    pass

class InformerVariableLengthDataModule(VariableLengthSequenceToLabelDataModule):
    pass