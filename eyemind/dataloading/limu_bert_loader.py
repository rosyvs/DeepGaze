# Based on code from Ekta Sood

from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from os.path import join
import numpy as np
import json
import pandas as pd
from pytorch_lightning import LightningDataModule
from eyemind.dataloading.transforms import Pooler
from functools import partial
def label_files(label_df,label_col,ids,id_col="filename"):
    labels = label_df[label_col].loc[label_df[id_col].isin(ids)].values.tolist()
    return labels
def create_filename_col(label_df, id_col="filename"):
    label_df[id_col] = label_df.apply(lambda row: f"{row['ParticipantID']}-{row['Text']}{str(row['PageNum']-1)}.csv", axis=1)
    return label_df

class GazeformerEmbeddingDataset(Dataset):
    def __init__(self, data_path, label_filepath,  min_sequence_length=125, max_sequence_length=125, label_col=None):
        self.data = np.load(data_path)
        self.dataset_name = data_path.split("/")[-1].split("_")[0]
        self.label_df = pd.read_csv(label_filepath, keep_default_na=True)
        self.label_id_col = "filename"
        self.label_df = create_filename_col(self.label_df, self.label_id_col)
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.label_col = label_col
        self.ids = self.filter_dataset()
        self.ixs = np.where([i["name"] in self.ids for i in self.data])[0]
        self.data = self.data[self.ixs]
        split = data_path.split("/")[-1].split("_")[2]
        self.fold=str(int(split)-1) # 0 indexed sorry
        self.label_df = create_filename_col(self.label_df, self.label_id_col)
        print(f"Dataset: {self.dataset_name}, label_col: {self.label_col}, fold: {self.fold}")
        print(f"len filtered data: {len(self.data)}")


    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        infos = self.data[idx]
        # print(infos['name'])
        embedding = torch.Tensor(infos["embedding"][:self.max_sequence_length, ...])  # out embedding from LIMU bertm [args.max_sequence_length, 72]
        # swap dims (feat_size, len)--> (len, feat_size) 
        embedding = embedding.permute(1, 0)
        ids = [infos['name']]
        og = infos['original_data']
        true_len = self.get_true_len(og)
        label = label_files(self.label_df, self.label_col, ids, self.label_id_col)
        return {"embedding":embedding, "sequence_label":label, 'true_len':true_len}

    def filter_dataset(self):
        label_col= self.label_col
        label_df = self.label_df
        if label_col:
            ids = label_df[(~label_df[label_col].isna()) & (label_df["sequence_length"] > self.min_sequence_length)][self.label_id_col].to_list()
        else:
            ids = label_df[id_col].loc[label_df["sequence_length"] > min_sequence_length].to_list()
        ids = set(ids)
        data_ids = set([i["name"] for i in self.data])
        return list(ids.intersection(data_ids))
        
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
        batch_tgt_y.append(t['sequence_label'])
        emb = t["embedding"]
        if pool_fn:
            emb = pool_fn(emb)
        batch_embedding.append(emb)
        pad_mask = torch.zeros(emb.shape[0])
        pad_mask[:true_len] = 1
        batch_pad_mask.append(pad_mask)
    batch_pad_mask = torch.stack(batch_pad_mask)
    batch_tgt_y = torch.Tensor(batch_tgt_y).float()
    batch_embedding = torch.stack(batch_embedding)
    return (batch_embedding, batch_pad_mask), batch_tgt_y

class EmbeddingDataModule(LightningDataModule):
    def __init__(self, train_data_path, label_filepath, batch_size=32, num_workers=4, pin_memory=True, min_sequence_length=125, max_sequence_length=125, label_col=None,
                 test_data_path=None, val_data_path=None, pool_method=None):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.label_filepath = label_filepath
        self.batch_size = batch_size
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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate_fn)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.collate_fn)

    # def get_collate_fn(self):
    #     return gazeformer_embedding_collate_fn(self.dataset)