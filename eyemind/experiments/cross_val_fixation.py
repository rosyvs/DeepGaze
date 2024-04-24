from argparse import ArgumentParser
import json
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
from eyemind.experiments.experimenter import BaseExperiment, Experiment
from eyemind.experiments.fixation_experiment import limit_label_seq
from eyemind.preprocessing.fixations import fixation_label_mapper
from eyemind.dataloading.load_dataset import limit_sequence_len
from eyemind.models.encoder_decoder import EncoderDecoderModel, load_encoder_decoder, create_encoder_decoder
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.model_selection import KFold
# Have to add path to enable torch.load to work since they saved it weirdly
import sys
sys.path.append(str(Path("../obf").resolve()))

from eyemind.obf.model import ae
from eyemind.obf.model import creator

def convert_to_float_tensor(data):
    return torch.tensor(data).float()
class Fixation_Experiment(Experiment):
    def __init__(self, config):
        self.class_weights = torch.tensor([3.,1.])
        super().__init__(config)

    def _split_fn(self, data):
        train_val, test = train_test_split(data, test_size=0.2, random_state=self.config.random_seed, shuffle=True)
        train, val = train_test_split(train_val, test_size=0.25, random_state=self.config.random_seed, shuffle=True)
        return train, val, test

    def setup_model(self):
        seq_lengths = self.config["sequence_length"]
        gru_hidden_size = self.config["gru_hidden_size"]
        criterion = nn.CrossEntropyLoss(self.class_weights)
        models = []
        for seq_length in seq_lengths:
            for size in gru_hidden_size:
                if self.config.pretrained_weights_dirpath:
                    encoder, decoder = load_encoder_decoder(self.config.pretrained_weights_dirpath, self.config.decoder_weights_filename)
                else:
                    encoder, decoder = create_encoder_decoder(hidden_dim=size, use_conv=self.config.use_conv, input_seq_length=seq_length)
                models.append(EncoderDecoderModel(encoder, decoder, criterion, 2, cuda=False, lr_scheduler_step_size=2))
        return models

    def get_transforms(self):
        seq_lengths = self.config["sequence_length"]
        x_transforms = []
        y_transforms = []
        for pc_seq_len in seq_lengths:
            lim_seq_len = partial(limit_sequence_len, sequence_len=pc_seq_len, random_part=False)
            limit_labels = partial(limit_label_seq, sequence_length=pc_seq_len) 
            x_transforms.append([lim_seq_len,convert_to_float_tensor])
            y_transforms.append([limit_labels,convert_to_float_tensor])
        return x_transforms, y_transforms
               
class Hyperparameter_Experiment(Experiment):
    def __init__(self, config):
        self.class_weights = torch.tensor([3.,1.])
        super().__init__(config)
        
    def setup(self):
        # Seed everything for reproducability
        pytorch_lightning.seed_everything(self.config.random_seed, workers=True)

        # Create datamodules
        self.label_mapper = self.get_label_map()
        self.transforms = self.get_transforms()
        self.data_modules = self.get_datamodules(self.label_mapper, *self.transforms)
        self.setup_datamodules()

        # Split Data
        self.data_splits = self.setup_data_splits()

        # Get Dataloaders
        self.dls = []
        for dm in self.data_modules:
            for _ in self.config.gru_hidden_size:
                self.dls.append(self.get_dataloaders(dm, self.data_splits))

        # Model
        self.models = self.setup_models()

        # Trainer
        self.trainers = self.setup_trainers()

    def setup_datamodules(self, stages=["fit"]):
        for stage in stages:
            for dm in self.data_modules:
                dm.setup(stage)

    def setup_models(self):
        seq_lengths = self.config.sequence_length
        gru_hidden_size = self.config.gru_hidden_size
        criterion = nn.CrossEntropyLoss(self.class_weights)
        models = []
        for seq_length in seq_lengths:
            for size in gru_hidden_size:
                if self.config.pretrained_weights_dirpath:
                    encoder, decoder = load_encoder_decoder(self.config.pretrained_weights_dirpath, self.config.decoder_weights_filename)
                else:
                    encoder, decoder = create_encoder_decoder(hidden_dim=size, use_conv=self.config.use_conv, input_seq_length=seq_length)
                models.append(EncoderDecoderModel(encoder, decoder, criterion, 2, cuda=False, lr_scheduler_step_size=2))
        return models

    def setup_trainers(self):
        trainers = []
        seq_lengths = self.config.sequence_length
        gru_hidden_size = self.config.gru_hidden_size      
        for seq_length in seq_lengths:
            for size in gru_hidden_size:
                logger=self.get_logger(sub_dir=f"pc_seq_len:{seq_length}-hidden_size:{size}")
                callbacks = self.get_callbacks()
                trainer = Trainer.from_argparse_args(self.config, logger=logger, callbacks=callbacks)
                trainers.append(trainer)
        return trainers

    def setup_data_splits(self):
        split_fn = self._split_fn()
        train_val, test = split_fn(np.arange(len(self.data_modules[0].dataset_train.files)), test_size=0.2, random_state=self.config.random_seed)
        train, val = split_fn(train_val, test_size=0.25, random_state=self.config.random_seed)
        return train, val, test

    def _split_fn(self):
        return train_test_split

    def get_transforms(self):
        seq_lengths = self.config.sequence_length
        x_transforms = []
        y_transforms = []
        for seq_len in seq_lengths:
            lim_seq_len = partial(limit_sequence_len, sequence_len=seq_len, random_part=False)
            limit_labels = partial(limit_label_seq, sequence_length=seq_len) 
            x_transforms.append([lim_seq_len,convert_to_float_tensor])
            y_transforms.append([limit_labels,convert_to_float_tensor])
        return x_transforms, y_transforms

    def get_datamodules(self, label_mapper, list_x_transforms, list_y_transforms):
        dms = [GazeDataModule(self.config.data_folderpath, label_mapper=label_mapper, transform_x=x_trans, transform_y=y_trans) for x_trans, y_trans in zip(list_x_transforms, list_y_transforms)]
        return dms

    def get_label_map(self):
        return fixation_label_mapper

    def run_all(self):
        for i in range(len(self.models)):
            train_dl, val_dl, _ = self.dls[i]
            model = self.models[i]
            trainer = self.trainers[i]
            print(self.run_fit(trainer, model, train_dl, val_dl))



def main(args):

    exp = Hyperparameter_Experiment(args)
    exp.run_all()


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Program Args
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--data_folderpath", type=str)
    parser.add_argument("--pretrained_weights_dirpath", type=str, default="")
    parser.add_argument("--decoder_weights_filename", type=str)
    parser.add_argument("--log_dirpath", type=str, default=".")
    parser.add_argument("--experiment_logging_folder", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--sequence_length", type=int, nargs="+")
    parser.add_argument("--stage", type=str, default="fit")
    parser.add_argument("--use_conv", type=bool, default=True)
    parser.add_argument("--gru_hidden_size", nargs="+", type=int)
    # add training arguments
    parser = Trainer.add_argparse_args(parser)

    # add model arguments

    args = parser.parse_args()

    # if there is a config path then load the arguments from that file
    if args.config_path:
        with open(Path(args.config_path).resolve(), 'r') as f:
            args.__dict__ = json.load(f)
    # no config path, so save the args to a file
    else:
        log_dirpath = Path(args.log_dirpath, args.experiment_logging_folder).resolve()
        log_dirpath.mkdir(parents=True, exist_ok=True)
        with open(Path(log_dirpath, "command_args.txt").resolve(), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    
    main(args)