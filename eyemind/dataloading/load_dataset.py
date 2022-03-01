from functools import partial
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

def label_files(label_df,id_col,label_col,filenames):
    # Strip extension
    ids = [f.split(".")[0] for f in filenames]
    #labels = label_df.loc[label_df[id_col].isin(ids)][label_col]
    labels = [label_df.loc[label_df[id_col] == id][label_col].values[0] for id in ids]
    return labels

def get_filenames_for_dataset(label_df,folder,id_col,label_col,ext="csv"):
    files = label_df[~label_df[label_col ].isna()][id_col].to_list()
    label_filenames = set([f"{file}.{ext}" for file in files])
    folder_filenames = set([f.name for f in folder.glob(f'*.{ext}')])
    return list(label_filenames.intersection(folder_filenames))

def get_id(row):
    return f"{row['ParticipantID']}-{row['Text']}{str(row['PageNum']-1)}"

def create_filename_col(label_df):
    label_df["filename"] = label_df.apply(lambda row: get_id(row), axis=1)
    return label_df

def get_label_mapper(label_df, id_col, label_col):
    #filenames = get_filenames_for_dataset(label_df, data_folder, id_col, label_col, ext)
    label_mapper = partial(label_files, label_df, id_col, label_col)
    return label_mapper
    
# What is a good method for choosing the sequence length?
def limit_sequence_len(x_data,sequence_len=3000,random_part=True):
    if len(x_data) > sequence_len:
        # Remove part data
        if random_part:
            start_idx = random.randint(0,len(x_data) - sequence_len)
            x_data = x_data[start_idx:start_idx+sequence_len]

    else:
        # Pad data
        padded_data = np.zeros((sequence_len,2))
        padded_data[:len(x_data)] = x_data
        x_data = padded_data
    return x_data

def get_samplers():
    pass

def get_stratified_group_splits(files, label_df, label_col, id_col, folds=4):
  enc = LabelEncoder()
  label_df = label_df[label_df[id_col].isin(files)]
  #label_df = label_df[~label_df[label_col].isna()]
  groups = enc.fit_transform(label_df[id_col].values)
  y = label_df[label_col]
  gkf = StratifiedGroupKFold(folds)
  splits = gkf.split(label_df,y,groups=groups)
  return splits