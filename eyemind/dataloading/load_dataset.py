from fileinput import filename
from functools import partial
from pathlib import Path
import random
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import torch
# from eyemind.dataloading.gaze_data import GazeDataModule
from torch.utils.data import SubsetRandomSampler
import yaml

def write_splits(splits, filepath, folds=False):
    if folds:
        split_out = {"folds": [{"train": split[0], "val": split[1]} for split in splits]}
    else:
        split_out = {"train": splits[0], "test": splits[1]}
    with open(filepath, 'w') as f:
        yaml.dump(split_out, f)
    return splits

def load_file_folds(path):
    with open(path, 'r') as f:
        file_folds_dict = yaml.safe_load(f)    
    return [(fold["train"], fold["val"]) for fold in file_folds_dict["folds"]]

def label_files(label_df,label_col,filenames,id_col="filename"):
    # Strip extension
    ids = [f.split(".")[0] for f in filenames]
    labels = label_df[label_col].loc[label_df[id_col].isin(ids)].values.tolist()
    #labels = [label_df.loc[label_df[id_col] == id][label_col].values[0] for id in ids]
    return labels

def get_filenames_for_dataset(folder, label_df, label_col, id_col="filename", ext="csv", min_sequence_length=500):
    files = label_df[(~label_df[label_col].isna()) & (label_df["sequence_length"] > min_sequence_length)][id_col].to_list()
    label_filenames = set([f"{file}.{ext}" for file in files])
    folder_filenames = set([f.name for f in Path(folder).glob(f'*.{ext}')])
    return list(label_filenames.intersection(folder_filenames))

def filter_files_by_seqlen(map_df, folder, min_sequence_length=500, ext="csv", id_col="filename"):
    folder_filenames = set([f.name for f in Path(folder).glob(f'*.{ext}')])
    files = map_df[id_col].loc[map_df["sequence_length"] > min_sequence_length].to_list()
    filenames = set([f"{file}.{ext}" for file in files])
    return list(filenames.intersection(folder_filenames))

def get_id(row):
    return f"{row['ParticipantID']}-{row['Text']}{str(row['PageNum']-1)}"

def get_seq_length(row, data_folder, id_col, ext):
    filename=row[id_col]
    path = Path(data_folder, f"{filename}.{ext}")
    try:
        df = pd.read_csv(path)
        sequence_length = len(df)
        return sequence_length
    except:
        return 0

def add_sequence_col(label_df, data_folder, id_col="filename", ext="csv"):
    label_df["sequence_length"] = label_df.apply(lambda row: get_seq_length(row, data_folder, id_col, ext), axis=1)
    return label_df

def create_filename_col(label_df):
    label_df["filename"] = label_df.apply(lambda row: get_id(row), axis=1)
    return label_df

def get_label_mapper(label_df, label_col):
    #filenames = get_filenames_for_dataset(label_df, data_folder, id_col, label_col, ext)
    label_mapper = partial(label_files, label_df, label_col, id_col="filename")
    return label_mapper

def get_label_df(label_path):
    label_df = pd.read_csv(label_path)
    label_df = create_filename_col(label_df)
    return label_df

# TODO: Write collate function which will return padded batch, sequence lengths, mask
def collate_fn_pad(batch, sequence_length):
    '''
    Pads batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    X, y = zip(*batch)
    X_lengths = torch.tensor([ t.shape[0] for t in X ])
    y_lengths = torch.tensor([ t.shape[0] for t in y ])
    ## padd
    X_padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-181.)
    y_padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-181.)
    # Split
    torch.split(X_padded, sequence_length, )
    ## compute mask
    return X_padded, y_padded, X_lengths, y_lengths

def split_collate_fn(sequence_length, batch):
    '''
    Takes variable length sequences, splits them each into 
    subsequences of sequence_length, and returns tensors:
    
    Args:
        batch: List[Tuples(Tensor(X), Tensor(y))] Contains a list of the returned items from dataset
        sequence_length: int lengths of the subsequences

    Returns:
        X: Tensor shape (bs, sequence_length, *)
        y: Tensor shape (bs, sequence_length)
    '''
    X, y = zip(*batch)
    # Splits each example into tensors with sequence length and drops last in case it is a different length
    X_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in X]
    y_splits = [torch.stack(torch.split(t, sequence_length, dim=0)[:-1], dim=0) for t in y]
    
    X = torch.cat(X_splits, dim=0)
    y = torch.cat(y_splits, dim=0)
    return X, y




# What is a good method for choosing the sequence length?
def limit_sequence_len(x_data,sequence_len=3000,random_part=True):
    if len(x_data) > sequence_len:
        # Remove part data
        if random_part:
            start_idx = random.randint(0,len(x_data) - sequence_len)
            x_data = x_data[start_idx:start_idx+sequence_len]
        else:
            x_data = x_data[:sequence_len]

    else:
        # Pad data
        padded_data = np.zeros((sequence_len,2))
        padded_data[:len(x_data)] = x_data
        x_data = padded_data
    return x_data

def get_samplers():
    pass

# def stratified_group_split(X, y, folds=4):
#     enc = LabelEncoder()
#     groups = enc.fit_transform(y)
#     gkf = StratifiedGroupKFold(folds)
#     splits = gkf.split(X, y, groups)
#     return splits

def get_stratified_group_splits(files, label_df, label_col, id_col="filename", group_col="ParticipantID", folds=4, seed=None):
    enc = LabelEncoder()
    files = [f.split(".")[0] for f in files]
    label_df = label_df[label_df[id_col].isin(files)]
    #label_df = label_df[~label_df[label_col].isna()]
    groups = enc.fit_transform(label_df[group_col].values)
    y = label_df[label_col]
    if seed:
        gkf = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    else:
        gkf = StratifiedGroupKFold(folds)
    splits = gkf.split(label_df,y,groups=groups)
    return splits

# def get_datamodule(label_col, label_df, data_folder, x_transforms=None, y_transforms=None, id_col="filename"):
#         filenames = get_filenames_for_dataset(label_df, data_folder, id_col, label_col)
#         label_mapper = get_label_mapper(label_df, id_col, label_col)
#         dm = GazeDataModule(data_folder, file_list=filenames, label_mapper=label_mapper, transform_x=x_transforms, transform_y=y_transforms)
#         return dm

def get_datamodules(label_cols, label_df, data_folder, x_transforms=None, y_transforms=None, id_col="filename"):
    l_ds = []
    for label_col in label_cols:
        dm = get_datamodule(label_col, label_df, data_folder, x_transforms=x_transforms, y_transforms=y_transforms, id_col=id_col)
        l_ds.append((label_col,dm))
    return l_ds

def get_dataloader_from_datamodule(dm, split=None, dl_type="train"):
    sampler=None
    if split:
        sampler = SubsetRandomSampler(split)
    if dl_type == "train":
        dl = dm.train_dataloader(sampler=sampler)
    elif dl_type == "val":
        dl = dm.val_dataloader(sampler=sampler)
    elif dl_type == "test":
        dl = dm.test_dataloader(sampler=sampler)
    else:
        raise ValueError("dl_type should be one of: train, val, test")
    return dl
