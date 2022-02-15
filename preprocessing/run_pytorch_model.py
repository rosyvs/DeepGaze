import os
from posixpath import split
import sys
import glob

from sklearn.model_selection import GroupKFold
#import dask.dataframe as dd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
#from data_utils import get_id
#from dask.distributed import Client
import pandas as pd
import numpy as np

# from load_data import CSVReader
# from preprocessing import ImputeMissingVals, FeatureScaler, EncodeLabels
# from pipeline import Pipeline
# from cross_validation import GenerateCVFolds, CrossValidationStage
# from model_init import ModelInitializer
# from evaluation_stage import EvaluationStage

def create_id_row(row):
    return f"{row['ParticipantID']}-{row['event']}"

def process_raw_data_pd(folder,label_filepath,label_col="Rote_X",ext="csv"):
    # Get all the x data in dask dataframe and group each data point
    l = []
    for f in folder.glob(f"*.{ext}"):
        l.append(pd.read_csv(f))
    x_df = pd.concat(l)
    print(x_df.head())
    x_df["id"] = x_df.apply(lambda row: f"{row['ParticipantID']}-{row['event']}",axis=1)
    x_df = x_df.drop(['t','event','ParticipantID'],axis=1)
    #xs_df = x_df.set_index('id')
    #print(xs_df)
    # Get the labels per data point
    label_df = pd.read_csv(label_filepath,usecols=['ParticipantID','Text','PageNum',label_col])
    label_df["id"] = label_df.apply(lambda row: get_id(row),axis=1)
    label_df = label_df[['id',label_col]]
    # Join data with labels
    #data_df = xs_df.merge(label_df,left_index=True, right_on="id",how="left")
    data_df = x_df.merge(label_df,on="id")
    # Remove NaNs
    data_df = data_df.loc[data_df[label_col].notnull()]
    data_df = data_df.set_index("id")
    return data_df

def process_sham_data_v2(folder, ext="csv"):
    pass

def process_sham_data(folder, ext="csv"):
    l = []
    for f in folder.glob(f"*.{ext}"):
        l.append(pd.read_csv(f))
    df = pd.concat(l)
    df['sham'] = df['event'].apply(lambda x: "Sham" in x)
    df = df.drop(["t"],axis=1)
    df = df.groupby(['ParticipantID', 'event']).agg(lambda x: list(x))
    df = df.reset_index()
    #print(df.head())
    # Could set column to have type object:
    df["scanpath"] = df.apply(lambda row: np.stack((np.array(row['XAvg']),np.array(row['YAvg'])),axis=1).astype(object),axis=1)
    #df["scanpath"] = df.apply(lambda row: [[row["XAvg"][i], row["YAvg"][i]] for i in range(len(row["XAvg"]))], axis=1)
    #df["scanpath"] = df["scanpath"].astype(object)
    df["sham"] = df["sham"].apply(lambda x: int(x[0]))
    print(df.head())
    df = df.loc[df["sham"].notnull()]
    df.drop(['XAvg','YAvg'],axis=1)
    return df

def preprocess_raw_data(folder,label_filepath,label_col="Rote_X",ext="csv"):
    # Get all the x data in dask dataframe and group each data point
    read_path = str(os.path.join(folder,f"*.{ext}"))
    print(read_path)
    x_df = dd.read_csv(read_path)
    print(x_df.head())
    x_df["id"] = x_df.apply(lambda row: f"{row['ParticipantID']}-{row['event']}",axis=1, meta=dd.utils.make_meta(str))
    x_df = x_df.drop(['t','event','ParticipantID'],axis=1)
    #xs_df = x_df.set_index('id')
    #print(xs_df)
    # Get the labels per data point
    label_df = dd.read_csv(label_filepath,usecols=['ParticipantID','Text','PageNum',label_col])
    label_df["id"] = label_df.apply(lambda row: get_id(row),axis=1, meta=dd.utils.make_meta(str))
    label_df = label_df[['id',label_col]]
    # Join data with labels
    #data_df = xs_df.merge(label_df,left_index=True, right_on="id",how="left")
    data_df = x_df.merge(label_df,on="id")
    # Remove NaNs
    data_df = data_df.loc[data_df[label_col].notnull()]
    data_df = data_df.set_index("id")
    return data_df


def get_subject_groups(df):
    
    subjects = df["ParticipantID"].unique()
    for i, id in enumerate(subjects):
        df.loc[df["ParticipantID"] == id, "groups"] = i
    print(df.head())
    #gkf = GroupKFold(folds)
    #gkf.split(df["scanpath"].to_numpy(), df["sham"].to_numpy(),groups=df["groups"].to_numpy())
    return df["groups"].to_numpy()

def test_process_raw_data(out_filename=""):
    folder = Path("/Users/rickgentry/emotive_lab/common-models-redesign/eyemind_data/scanpaths").resolve()
    label_filepath = Path("/Users/rickgentry/emotive_lab/common-models-redesign/eyemind_data/labels/EML1_pageLevel.csv").resolve()
    data_df = process_raw_data_pd(folder,label_filepath)
    print(data_df.head())
    if out_filename:
        data_df.to_csv(out_filename)


def test_preprocess_raw_data(out_filename=""):
    folder = Path("/Users/rickgentry/emotive_lab/common-models-redesign/eyemind_data/small_data").resolve()
    label_filepath = Path("/Users/rickgentry/emotive_lab/common-models-redesign/eyemind_data/labels/EML1_pageLevel.csv").resolve()
    data_df = preprocess_raw_data(folder,label_filepath)
    print(data_df.head())
    if out_filename:
        dd.to_csv(data_df,out_filename)

def test_process_sham_data(out_filepath=""):
    folder = Path("/Users/rickgentry/emotive_lab/eyemind/data/preprocessed/shamnosham_output")
    df = process_sham_data(folder)
    if out_filepath:
        df.to_csv(out_filepath,index=False)

if __name__ == "__main__":
    #test_preprocess_raw_data(Path("/Users/rickgentry/emotive_lab/common-models-redesign/eyemind_data/combined/scanpaths-*.csv"))
    df = process_sham_data(Path("/Users/rickgentry/emotive_lab/eyemind/data/preprocessed/shamnosham_output"))
    #test_process_sham_data(out_filepath=Path("/Users/rickgentry/emotive_lab/eyemind/data/processed", "shamnosham.csv"))
    groups = get_subject_groups(df)
    gkf = GroupKFold(4)
    X = df["scanpath"].to_numpy()
    y = df["sham"].to_numpy()
    for train_index, test_index in gkf.split(X, y, groups):
        print("TRAIN:", train_index, "TEST:", test_index)
        assert(len(np.intersect1d(train_index,test_index))==0)
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        # print(X_train, X_test, y_train, y_test)

 
