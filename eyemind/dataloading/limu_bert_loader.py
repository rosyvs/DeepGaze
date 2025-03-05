# Based on code from Ekta Sood

from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import re
from os.path import join
import numpy as np
import json
import pandas as pd
from pytorch_lightning import LightningDataModule
from eyemind.dataloading.transforms import Pooler
from functools import partial

def label_files(label_df,label_col,id,id_col="filename"):
    label = label_df[label_col].loc[label_df[id_col]==id].values
    if len(label)>1:
        # needs to be an array of one element in float64
        label = np.array([label[0]])    # to torch array
    label = torch.tensor(label).float()
    return label

def create_filename_col(label_df, id_col="filename"):
    label_df[id_col] = label_df.apply(lambda row: f"{row['ParticipantID']}-{row['Text']}{str(row['PageNum']-1)}.csv", axis=1)
    return label_df

def get_split_from_datapath(data_path):
    data_path = str(data_path)
    try:
        split = data_path.split("/")[-1].split("_")[2]
        # check if isplit is a number
        int(split)
        print(f"split extracted from filename: {split}")
        return split
    except:
        try:
            # search for substring "split" followed by an integer
            split = re.search(r'split(\d+)', data_path).group(1)
            int(split)
            print(split)
            print(f"split extracted from filename: {split}")
            return split
        except:
            print(f"Could not find split number in data path: {data_path}")
            return None

class GazeformerEmbeddingDataset(Dataset):
    def __init__(self, data_path, label_filepath,  min_sequence_length=125, max_sequence_length=125, label_col=None):
        self.data = np.load(data_path)
        # print(f"Data has length: {len(self.data)}")
        self.dataset_name = data_path.split("/")[-1].split("_")[0]
        # print(f"Dataset name extracted from data path: {self.dataset_name}")
        self.label_df = pd.read_csv(label_filepath, keep_default_na=True)
        self.label_id_col = "filename"
        self.label_df = create_filename_col(self.label_df, self.label_id_col)
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.label_col = label_col
        # print(f"Label column: {self.label_col}")
        self.ids = self.filter_dataset()
        # print(f"len filtered data: {len(self.ids)}")
        self.ixs = np.where([i["name"] in self.ids for i in self.data])[0]
        # print(f"len ixs: {len(self.ixs)}")
        self.data = self.data[self.ixs]
        split = get_split_from_datapath(data_path)
        self.fold=str(int(split)-1) # 0 indexed sorry
        self.label_df = create_filename_col(self.label_df, self.label_id_col)
        # print(f"label_df: {self.label_df.columns}, length: {len(self.label_df)}")
        # print(f"Dataset: {self.dataset_name}, label_col: {self.label_col}, fold: {self.fold}")
        # print(f"len filtered data: {len(self.data)}")


    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        infos = self.data[idx]
        # print(infos['name'])
        embedding = torch.Tensor(infos["embedding"][:self.max_sequence_length, ...])  # out embedding from LIMU bertm [args.max_sequence_length, 72]
        id = infos['name']
        og = infos['embedding']
        true_len = self.get_true_len(og)
        label = label_files(self.label_df, self.label_col, id, self.label_id_col)
        return {"embedding":embedding, "sequence_label":label, 'true_len':true_len}

    def filter_dataset(self):
        label_col= self.label_col
        label_df = self.label_df
        # print(f"Filter dataset on label_col: {label_col}")
        # print(f"...using label df: {label_df.columns}, length: {len(label_df)}")
        # fitler on labeldf stuff
        if label_col:
            # count na in label col
            # print(f"sequences with NaN in label col: {label_df[label_col].isna().sum()}")
            # print(f"sequences with length less than {self.min_sequence_length}: {len(label_df[label_df['sequence_length']<self.min_sequence_length])}")
            ids = label_df[(~label_df[label_col].isna())][self.label_id_col].to_list()
        else:
            # print(f"sequences with length less than {self.min_sequence_length}: {len(label_df[label_df['sequence_length']<self.min_sequence_length])}")

            ids = label_df[id_col].to_list()
        
        ids = set(ids)
        print(f"len ids: {len(ids)}")
        # print(f"random sample of ids: {list(ids)[:5]}")
        # fitler on data true_len
        true_lens = [self.get_true_len(i["embedding"]) for i in self.data]
        print(f"sequences with length less than {self.min_sequence_length}: {len([i for i in true_lens if i<self.min_sequence_length])}")
        # print(f"sequences with length less than {self.min_sequence_length}: {len([i for i in true_lens if i<self.min_sequence_length])}")
        data_ids = set([i["name"] for i in self.data if self.get_true_len(i["embedding"])>=self.min_sequence_length])
        print(f"len data_ids: {len(data_ids)}")
        # print(f"random sample of data_ids: {list(data_ids)[:5]}")'
        sel = list(ids.intersection(data_ids))
        return sel
        
    def get_true_len(self, xi, pad_val=-1.0):
        # len, 3
        # length of sequence is up to where no value is equal to pad val in the 2nd dim
        true_ix=np.where((xi!=pad_val).sum(axis=1)==xi.shape[1])
        true_length = true_ix[0][-1]+1
        return true_length 

def gazeformer_embedding_collate_fn(batch, pool_fn=None): 
    # takes batched output of getitem and makes a batch in the form of a tuple of ((X, Xmask), y) 
    # as this is the format expected by the model classes
    batch_tgt_y = []
    batch_embedding = []
    batch_pad_mask = [] 
    # print(batch)
    for t in batch:
        true_len = t['true_len']
        y = t['sequence_label']
        if len(y)>1:
            raise ValueError("More than one label found for a sample")
        batch_tgt_y.append(y)
        emb = t["embedding"] 
        batch_embedding.append(emb)
        pad_mask = torch.zeros(emb.shape[0])
        pad_mask[:true_len] = 1
        batch_pad_mask.append(pad_mask)
    batch_pad_mask = torch.stack(batch_pad_mask)
    batch_tgt_y = torch.Tensor(batch_tgt_y).float()
    batch_embedding = torch.stack(batch_embedding)
    if pool_fn:
        batch_embedding = pool_fn(batch_embedding, batch_pad_mask)
        batch_pad_mask = None #because no longer a time dim
    return (batch_embedding, batch_pad_mask), batch_tgt_y

class EmbeddingDataModule(LightningDataModule):
    def __init__(self, train_data_path, label_filepath, batch_size=32, num_workers=4, pin_memory=True, min_sequence_length=2, max_sequence_length=125, label_col=None,
                 test_data_path=None, val_data_path=None, pool_method=None, drop_last=True):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.label_filepath = label_filepath
        self.batch_size = batch_size
        self.drop_last= drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.label_col = label_col
        self.pool_method = pool_method
        self.pool_fn = Pooler(pool_method)


    def setup(self, stage=None):
        self.train_dataset = GazeformerEmbeddingDataset(self.train_data_path, self.label_filepath, label_col=self.label_col, max_sequence_length=self.max_sequence_length)
        self.val_dataset = GazeformerEmbeddingDataset(self.val_data_path, self.label_filepath, label_col=self.label_col, max_sequence_length=self.max_sequence_length)
        if self.test_data_path:
            self.test_dataset = GazeformerEmbeddingDataset(self.test_data_path, self.label_filepath, label_col=self.label_col, max_sequence_length=self.max_sequence_length)
        self.collate_fn = partial(gazeformer_embedding_collate_fn, pool_fn=self.pool_fn)
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate_fn, drop_last=self.drop_last)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate_fn, drop_last=self.drop_last)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate_fn, drop_last=self.drop_last)

    # def get_collate_fn(self):
    #     return gazeformer_embedding_collate_fn(self.dataset)