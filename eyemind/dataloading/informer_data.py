from functools import partial
import random
from typing import List, Optional, Union
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from eyemind.dataloading.gaze_data import BaseGazeDataModule, BaseSequenceToSequenceDataModule, GroupStratifiedKFoldDataModule, SequenceLabelDataset, SequenceToSequenceDataModule
from torch.utils.data import Dataset, DataLoader, Subset

from eyemind.dataloading.load_dataset import filter_files_by_seqlen, get_label_df
from eyemind.dataloading.batch_loading import *
from eyemind.dataloading.transforms import ToTensor
from eyemind.preprocessing.fixations import fixation_label_mapper


class InformerDataModule(BaseSequenceToSequenceDataModule):

    def __init__(self,
                data_dir: str,
                label_filepath: str,
                load_setup_path: Optional[str] = None,
                test_dir: Optional[str] = None,
                train_dataset: Optional[Dataset] = None,
                val_dataset: Optional[Dataset] = None,
                test_dataset: Optional[Dataset] = None,
                train_fold: Optional[Dataset] = None,
                val_fold: Optional[Dataset] = None,
                sequence_length: int = 250,
                label_length: int = 48,
                pred_length: Optional[int] = None,
                num_workers: int = 0,
                batch_size: int = 8,
                pin_memory: bool = True,
                drop_last: bool = True,
                min_scanpath_length: int = 500,
                ):
        super().__init__(data_dir,
                        label_filepath,
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
                        min_scanpath_length
                        )
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.pred_length = pred_length
        self.label_length = label_length


    def prepare_data(self):
        '''
        Download and save data, do some preprocessing before transforms are applited
        '''
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", "predict", None):
            dataset = SequenceLabelDataset(self.data_dir, file_mapper=self.file_mapper, label_mapper=self.label_mapper, transform_x=self.x_transforms, transform_y=self.y_transforms, usecols=[2,3], scale=True)
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
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last, 
            pin_memory=self.pin_memory,
            collate_fn=partial(random_collate_fn, self.sequence_length))
    
    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        group = parent_parser.add_argument_group("InformerDataModule")
        group.add_argument("--data_dir", type=str)
        group.add_argument("--test_dir", type=str, default="")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--label_filepath", type=str)
        group.add_argument("--sequence_length", type=int, default=250)
        group.add_argument("--min_scanpath_length", type=int, default=500)
        return parent_parser

    @property
    def x_transforms(self):
        return ToTensor()

    @property
    def y_transforms(self):
        return ToTensor()

    @property
    def label_mapper(self):
        return partial(fixation_label_mapper, self.data_dir)
    
    @property
    def file_mapper(self):
        return partial(filter_files_by_seqlen, self.label_df, min_sequence_length=self.min_scanpath_length)

def informer_collate(sequence_length, pred_length, label_length, batch):
    '''
    Returns batches for 4 tasks. FixationID, Predictive Coding, Reconstruction, Contrastive Learning
    
    Args:
        sequence_length: (int) Length of encoder input
        pred_length: (int) Length of predictions for decoder output
        label_length: (int) Length of input labels to decoder for start
        batch: (List[Tuples(Tensor(X), Tensor(y))]) Contains a list of the returned items from dataset
    '''
    
    X, fixation_labels = zip(*batch)
    
    # 1. Fixation ID 
    # 2. Reconstruction (RC) (Just uses same X batch and the y is made from the X)
    fixation_decoder_inp, fixation_labels = fixation_batch(sequence_length,label_length, pred_length, X, fixation_labels)
    rc_decoder_inp = reconstruction_batch(X, label_length)
    # 3. Predictive Coding (PC)
    X_pc, pc_labels = predictive_coding_batch(X, sequence_length, pred_length, label_length)

    # 4. Contrastive Learning (CL)
    X_cl1, X_cl2, cl_labels = contrastive_batch(X, sequence_length)

    return X, fixation_decoder_inp, fixation_labels, rc_decoder_inp, X_pc, pc_labels, X_cl1, X_cl2, cl_labels
