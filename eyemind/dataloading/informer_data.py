from functools import partial
import random
from typing import List, Optional, Union

import torch
from eyemind.dataloading.gaze_data import BaseGazeDataModule, BaseSequenceToSequenceDataModule, GroupStratifiedKFoldDataModule, SequenceLabelDataset, SequenceToSequenceDataModule
from torch.utils.data import Dataset, DataLoader, Subset

from eyemind.dataloading.load_dataset import filter_files_by_seqlen, get_label_df, split_collate_fn
from eyemind.dataloading.transforms import ToTensor
from eyemind.preprocessing.fixations import fixation_label_mapper


class InformerDataModule(BaseSequenceToSequenceDataModule):

    def __init__(self,
                data_dir: str,
                label_filepath: str,
                load_setup_path: Optional[str] = None,
                test_dir: Optional[str] = None,
                train_dataset: Optional[Dataset] = None,
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
                ):
        super().__init__(data_dir,
                        label_filepath,
                        load_setup_path,
                        test_dir,
                        train_dataset,
                        test_dataset,
                        sequence_length,
                        num_workers,
                        batch_size,
                        pin_memory,
                        drop_last)
        self.pred_length = pred_length
        self.label_length = label_length


    def prepare_data(self):
        '''
        Download and save data, do some preprocessing before transforms are applited
        '''
        pass

            
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
        group = parent_parser.add_argument_group("InformerDataModule")
        group.add_argument("--data_dir", type=str)
        group.add_argument("--test_dir", type=str, default="")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--label_filepath", type=str)
        group.add_argument("--sequence_length", type=int, default=250)
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
        return partial(filter_files_by_seqlen, self.label_df)

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

def fixation_batch(input_length, label_length, pred_length, X, y, padding=0.):
    '''
    Takes variable length sequences, splits them each into 
    subsequences of sequence_length, and returns tensors

    Args:
        input_length: (int) Length of encoder input
        pred_length: (int) Length of predictions for decoder output
        label_length: (int) Length of input labels to decoder for start
        X: (Tensor) Sequence data (bs, sequence_length_scanpath, 2)
        y: (Tensor) Fixation labels (bs, sequence_length_scanpath)
    
    Returns:
        X: Gaze data (bs*sequence_length_scanpath/input_length, input_length, 2)
        decoder_inp: Input to incoder with first label_length of each batch as the actual fixation data and the rest masked 
        targets: Fixation labels for each subsequence
    '''

    targets = y[:, -pred_length:]
    if padding == 0:
        decoder_inp = torch.zeros_like(y)
    else: 
        decoder_inp = torch.ones_like(y) * padding
    decoder_inp[:, :label_length] = y[:,:label_length]
    return decoder_inp, targets    

    
def predictive_coding_batch(X_batch, input_length, pred_length, label_length):
    # total_len = input_length + pred_length
    # X_split = [torch.stack(torch.split(t, total_len, dim=0)[:-1], dim=0) for t in X_batch]
    X_seq = X_batch[:,:input_length,:]
    y_seq = X_batch[:,input_length-label_length:input_length+pred_length,:]
    # print(X_seq.shape, y_seq.shape)
    # X = torch.cat(X_seq, dim=0)
    # y = torch.cat(y_seq, dim=0)
    return X_seq, y_seq

def reconstruction_batch(X_batch, label_length):
    decoder_inp = torch.zeros_like(X_batch)
    decoder_inp[:,:label_length,:] = X_batch[:,:label_length,:]
    return decoder_inp

def contrastive_batch(X_batch, input_length):
    n,sl,fs = X_batch.shape
    x1 = torch.zeros((n, input_length, fs))
    x2 = torch.zeros((n, input_length, fs))
    y = torch.zeros(n)
    for i in range(n):
    # get x1
        try:
            x1_start = random.randrange(0, sl - input_length)
        except:
            x1_start = 0
        x1[i, :, :] = X_batch[i, x1_start:x1_start + input_length, :]

        if random.random() > 0.5:
            # Get x2 from the same sequence
            j = i
            y[i] = 1
        else:
            # Get x2 from different sequence
            j = i
            y[i] = 0
            while j == i:
                j = random.randrange(0, n)

        try:
            x2_start = random.randrange(0, sl - input_length)
        except:
            x2_start = 0

        x2[i, :, :] = X_batch[j, x2_start:x2_start + input_length, :]

    return x1.float(), x2.float(), y.float()


