from argparse import ArgumentParser
from distutils.command.config import config
import json
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import SubsetRandomSampler
import torch
from torch import nn
from .eyemind.dataloading.gaze_data import BaseSequenceToSequenceDataModule, GazeDataModule, SequenceToSequenceDataModule
from .eyemind.experiments.cli import GazeLightningCLI
from .eyemind.experiments.experimenter import BaseExperiment
from .eyemind.models.transformers import InformerEncoderDecoderModel
from .eyemind.preprocessing.fixations import fixation_label_mapper
from .eyemind.dataloading.load_dataset import limit_sequence_len
from .eyemind.models.encoder_decoder import EncoderDecoderModel, VariableSequenceLengthEncoderDecoderModel
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import LearningRateMonitor
# Have to add path to enable torch.load to work since they saved it weirdly
import sys
sys.path.append(str(Path("../obf").resolve()))

from .eyemind.obf.model import ae
from .eyemind.obf.model import creator

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

def load_encoder_decoder(pretrained_weights_dirpath, decoder_weights_filename):
    encoder = creator.load_encoder(str(Path(pretrained_weights_dirpath).resolve()))
    decoder = torch.load(str(Path(pretrained_weights_dirpath, decoder_weights_filename).resolve()),map_location=torch.device('cpu'))
    return encoder, decoder

def create_encoder_decoder(hidden_dim=128, use_conv=True, conv_dim=32, input_dim=2, out_dim=2, input_seq_length=500, backbone_type="gru", nlayers=2):
    if use_conv:
        enc_layers = [
            ae.CNNEncoder(input_dim=input_dim, latent_dim=conv_dim, layers=[
                16,
            ]),
            ae.RNNEncoder(input_dim=conv_dim,
                        latent_dim=hidden_dim,
                        backbone=backbone_type,
                        nlayers=nlayers,
                        layer_norm=False)
        ] 
    else: 
        enc_layers = [ae.RNNEncoder(input_dim=input_dim,
                        latent_dim=hidden_dim,
                        backbone=backbone_type,
                        nlayers=nlayers,
                        layer_norm=False)]

    encoder = nn.Sequential(*enc_layers)

    fi_decoder = ae.RNNDecoder(input_dim=hidden_dim,
                               latent_dim=hidden_dim,
                               out_dim=out_dim,
                               seq_length=input_seq_length,
                               backbone=backbone_type,
                               nlayers=nlayers,
                               batch_norm=True)

    return encoder, fi_decoder

def setup_data_module(data_folderpath, sequence_length, label_mapper, stage="fit"):
    lim_seq_len = partial(limit_sequence_len, sequence_len=sequence_length, random_part=False)
    limit_labels = partial(limit_label_seq, sequence_length=sequence_length)
    transforms = [lim_seq_len,lambda data: torch.tensor(data).float()]
    dm = GazeDataModule(data_folderpath, label_mapper=label_mapper, transform_x=transforms, transform_y=[limit_labels,lambda data: torch.tensor(data).float()], num_workers=0)
    dm.setup(stage)
    return dm

def find_lr():
    # Random Seed
    seed = 42
    pytorch_lightning.seed_everything(seed, workers=True)    
    
    # Data Module Creation
    data_folder = Path("./data/processed/fixation")
    sequence_len = 500
    lim_seq_len = partial(limit_sequence_len, sequence_len=sequence_len, random_part=False)
    limit_labels = partial(limit_label_seq, sequence_length=sequence_len)
    transforms = [lim_seq_len,lambda data: torch.tensor(data).float()]
    # Setting num workers throws an error because of trying to pickle lambda functions
    dm = GazeDataModule(data_folder, label_mapper=fixation_label_mapper, transform_x=transforms, transform_y=[limit_labels,lambda data: torch.tensor(data).float()], num_workers=0)
    dm.setup('fit')

    # Split Data
    train_split, test_split = train_test_split(np.arange(len(dm.dataset_train.files)), test_size=0.2, random_state=seed, shuffle=True)
    train_test_splits = [(train_split, test_split)]
    # Get dataloaders
    train_dl, val_dl = get_dataloaders_from_split(dm, train_split, test_split)

    # Model
    pre_trained_weights_dir = Path("./OBF/pre_weights/sample_weights")
    encoder = creator.load_encoder(str(pre_trained_weights_dir.resolve()))
    fi_decoder = torch.load(str(Path(pre_trained_weights_dir, "fi_1633040995_gru.pt").resolve()),map_location=torch.device('cpu'))
    class_weights = torch.tensor([3.86, 0.26])
    criterion = nn.CrossEntropyLoss(class_weights/ class_weights.sum())
    fi_encoder_decoder_model = EncoderDecoderModel(encoder, fi_decoder, criterion, 2, cuda=False)

    # Tuner
    trainer = Trainer()
    lr_finder = trainer.tuner.lr_find(fi_encoder_decoder_model, train_dl, val_dl)
    #print(lr_finder.results)
    fig = lr_finder.plot(suggest=True, show=True)
    new_lr = lr_finder.suggestion()
    print(new_lr)

def train_from_scratch(use_conv=True, tune=False):

    # Random Seed
    seed = 42
    pytorch_lightning.seed_everything(seed, workers=True)
    # Data Module Creation
    data_folder = Path("./data/processed/fixation")
    sequence_len = 500
    lim_seq_len = partial(limit_sequence_len, sequence_len=sequence_len, random_part=False)
    limit_labels = partial(limit_label_seq, sequence_length=sequence_len)
    transforms = [lim_seq_len,lambda data: torch.tensor(data).float()]
    # Setting num workers throws an error because of trying to pickle lambda functions
    dm = GazeDataModule(data_folder, label_mapper=fixation_label_mapper, transform_x=transforms, transform_y=[limit_labels,lambda data: torch.tensor(data).float()], num_workers=0)
    dm.setup('fit')

    # Split Data
    train_split, test_split = train_test_split(np.arange(len(dm.dataset_train.files)), test_size=0.2, random_state=seed, shuffle=True)
    train_test_splits = [(train_split, test_split)]

    # Model
    encoder, fi_decoder = create_encoder_decoder(use_conv=use_conv)
    class_weights = torch.tensor([3.86, 0.26])
    criterion = nn.CrossEntropyLoss(class_weights/ class_weights.sum())
    fi_encoder_decoder_model = EncoderDecoderModel(encoder, fi_decoder, criterion, 2, cuda=False)


    if tune:
        train_dl, val_dl = get_dataloaders_from_split(dm, train_split, test_split)
        trainer = Trainer()
        lr_finder = trainer.tuner.lr_find(fi_encoder_decoder_model, train_dl, val_dl)
        #print(lr_finder.results)
        fig = lr_finder.plot(suggest=True, show=True)
    else:

        # Trainer
        grad_norm = 0.5
        logger = TensorBoardLogger("lightning_logs", name="fixation_id_conv_not_pretrained")
        trainer = Trainer(max_epochs=1, logger=logger, deterministic=True, gradient_clip_val=grad_norm)

        # Set up experiment
        experiment = BaseExperiment(dm, [fi_encoder_decoder_model], [trainer], train_val_splits=train_test_splits)
        experiment.run_cross_val()


def main(args):

    # Random Seed
    seed = args.random_seed
    pytorch_lightning.seed_everything(seed, workers=True)

    # Data Module Creation
    dm = setup_data_module(args.data_folderpath, args.sequence_length, fixation_label_mapper, args.stage)

    # Split Data
    train_split, test_split = train_test_split(np.arange(len(dm.dataset_train.files)), test_size=0.2, random_state=seed, shuffle=True)
    train_test_splits = [(train_split, test_split)]

    # Model
    if args.pretrained_weights_dirpath:
        encoder, decoder = load_encoder_decoder(args.pretrained_weights_dirpath, args.decoder_weights_filename)
    else:
        encoder, decoder = create_encoder_decoder(use_conv=args.use_conv, input_seq_length=args.sequence_length)
    #class_weights = torch.tensor([3.86, 0.26])
    class_weights = torch.tensor([3., 1.])
    #criterion = nn.CrossEntropyLoss(class_weights/ class_weights.sum())
    criterion = nn.CrossEntropyLoss(class_weights)
    fi_encoder_decoder_model = EncoderDecoderModel(encoder, decoder, criterion, 2, cuda=False, lr_scheduler_step_size=2)

    # Trainer
    #grad_norm = 0.5
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(args.log_dirpath, name=args.experiment_logging_folder)
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=[lr_monitor])

    # Set up experiment
    experiment = BaseExperiment(dm, [fi_encoder_decoder_model], [trainer], train_val_splits=train_test_splits)
    last_epoch_metrics = experiment.run_cross_val()
    print(last_epoch_metrics)


if __name__ == "__main__":
    cli = LightningCLI(VariableSequenceLengthEncoderDecoderModel, BaseSequenceToSequenceDataModule, seed_everything_default=42, trainer_defaults={'max_epochs': 5})
    #cli = LightningCLI(InformerEncoderDecoderModel, BaseSequenceToSequenceDataModule, seed_everything_default=42, trainer_defaults={'max_epochs': 5, 'num_sanity_val_steps': 0})