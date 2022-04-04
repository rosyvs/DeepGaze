from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import SubsetRandomSampler
import torch
from torch import nn
from eyemind.dataloading.gaze_data import GazeDataModule
from eyemind.experiments.experimenter import BaseExperiment
from eyemind.preprocessing.fixations import fixation_label_mapper
from eyemind.dataloading.load_dataset import limit_sequence_len
from eyemind.models.encoder_decoder import EncoderDecoderModel
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path("/Users/rickgentry/emotive_lab/eyemind/OBF").resolve()))
from obf.model import ae
from obf.model import creator


def limit_label_seq(y_data, sequence_length, pad_token=-1.):
    if len(y_data) > sequence_length:
        y_data = y_data[:sequence_length]
    else:
        pad_data = np.ones((sequence_length,)) * pad_token
        pad_data[:len(y_data)] = y_data
        y_data = pad_data
    return y_data

def get_dataloaders_from_split(dm, train_split, val_split):
    train_sampler = SubsetRandomSampler(train_split)
    train_dl = dm.train_dataloader(sampler=train_sampler)
    val_sampler = SubsetRandomSampler(val_split)
    val_dl = dm.val_dataloader(sampler=val_sampler)
    return train_dl, val_dl

def main():

    # Random Seed
    seed = 42
    pytorch_lightning.seed_everything(seed, workers=True)
    # Data Module Creation
    data_folder = Path("/Users/rickgentry/emotive_lab/eyemind/data/processed/fixation")
    sequence_len = 500
    lim_seq_len = partial(limit_sequence_len, sequence_len=sequence_len, random_part=False)
    limit_labels = partial(limit_label_seq, sequence_length=sequence_len)
    transforms = [lim_seq_len,lambda data: torch.tensor(data).float()]
    dm = GazeDataModule(data_folder, label_mapper=fixation_label_mapper, transform_x=transforms, transform_y=[limit_labels,lambda data: torch.tensor(data).float()], num_workers=5)
    dm.setup('fit')

    # Split Data
    train_split, test_split = train_test_split(np.arange(len(dm.dataset_train.files)), test_size=0.2, random_state=seed, shuffle=True)
    train_test_splits = [(train_split, test_split)]

    # Model
    pre_trained_weights_dir = Path("/Users/rickgentry/emotive_lab/eyemind/OBF/pre_weights/sample_weights")
    encoder = creator.load_encoder(str(pre_trained_weights_dir.resolve()))
    fi_decoder = torch.load(str(Path(pre_trained_weights_dir, "fi_1633040995_gru.pt").resolve()),map_location=torch.device('cpu'))
    class_weights = torch.tensor([3.86, 0.26])
    criterion = nn.CrossEntropyLoss(class_weights/ class_weights.sum())
    fi_encoder_decoder_model = EncoderDecoderModel(encoder, fi_decoder, criterion, 2, cuda=False)

    # Trainer
    logger = TensorBoardLogger("../../lightning_logs", name="fixation_id_full_train")
    trainer = Trainer(max_epochs=5, logger=logger, deterministic=True)

    # Set up experiment
    experiment = BaseExperiment(dm, [fi_encoder_decoder_model], [trainer], train_val_splits=train_test_splits)
    experiment.run_cross_val()


if __name__ == "__main__":
    #parser = ArgumentParser()
    
    # Experiment Args

    # add training arguments
    #parser = Trainer.add_argparse_args(parser)

    # add model arguments


    # add data module args
    main()