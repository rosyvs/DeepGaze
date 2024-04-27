from functools import partial
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
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

def label_files(label_df,label_col,filenames,id_col="filename"):
    ids = [f.split(".")[0] for f in filenames]    # Strip extension
    labels = label_df[label_col].loc[label_df[id_col].isin(ids)].values.tolist()
    return labels

def label_samples(folder, filenames, label_col='fixation_label'):
    labels = []
    for f in filenames:
        try:
            df = pd.read_csv(str(Path(folder,f).resolve()))
        except Exception as e:
            print(f'{str(Path(folder,f).resolve())}')
            raise e
        label_array = df[label_col].to_numpy(float)
        labels.append(label_array)
    return labels

def label_samples_and_files(folder, filenames, label_df, sample_label_col='fixation_label',file_label_col='readWPM', id_col="filename"):
    # labeller to get sequence-(file)-level labels AND sample-level labels
    ids = [f.split(".")[0] for f in filenames]    # Strip extension
    sample_labels = []
    for f in filenames:
        try:
            df = pd.read_csv(str(Path(folder,f).resolve()))
        except Exception as e:
            print(f'{str(Path(folder,f).resolve())}')
            raise e
        label_array = df[sample_label_col].to_numpy(float)
        sample_labels.append(label_array)
    if file_label_col:
        file_labels = label_df[file_label_col].loc[label_df[id_col].isin(ids)].values.tolist()
        labels = list(zip(sample_labels, file_labels))
    else:
        return sample_labels
    return labels

def get_label_mapper(label_df, label_col): # this only works for file level labels
    label_mapper = partial(label_files, label_df, label_col, id_col="filename")
    return label_mapper

def get_label_df(label_path):
    label_df = pd.read_csv(label_path)
    return label_df


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

def binarize_labels(y_data, threshold=0.5):
    # generate 0 if below threshold, 1 if above
    y_data = (y_data > threshold)
    return y_data

# def stratified_group_split(X, y, folds=4):
#     enc = LabelEncoder()
#     groups = enc.fit_transform(y)
#     gkf = StratifiedGroupKFold(folds)
#     splits = gkf.split(X, y, groups)
#     return splits

# Implement splitting without stratification
def get_participant_splits(files, label_df, id_col="filename", group_col="ParticipantID", folds=4, seed=None):
    enc = LabelEncoder()
    files = [f.split(".")[0] for f in files]
    files_partids = [f.split("-")[0] for f in files]
    #label_df = label_df[label_df[id_col].isin(files)]
    #groups = enc.fit_transform(label_df[group_col].values)
    groups = enc.fit_transform(files_partids)
    #y = label_df[label_col]
    if seed:
        gkf = GroupKFold(folds, shuffle=True, random_state=seed)
    else:
        gkf = GroupKFold(folds)
    #splits = gkf.split(label_df,y,groups=groups)
    splits = gkf.split(files,groups=groups)
    return splits

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
