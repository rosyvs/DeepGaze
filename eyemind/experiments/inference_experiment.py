from argparse import ArgumentParser
from email import parser
from functools import partial
import json
from pathlib import Path
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning
from pytorch_lightning.utilities.cli import LightningCLI
from sklearn.model_selection import train_test_split
from eyemind.dataloading.gaze_data import GazeDataModule, SequenceToLabelDataModule
from eyemind.dataloading.load_dataset import get_filenames_for_dataset, get_stratified_group_splits
from eyemind.experiments.experimenter import Experiment
from eyemind.models.classifier import EncoderClassifierModel

class InferenceExperiment(Experiment):

    splitters = {"train_test_split": train_test_split,
                 "stratified_group_kfold": get_stratified_group_splits,
                }


    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        # Seed everything for reproducability
        pytorch_lightning.seed_everything(self.config.random_seed, workers=True)

        # Create datamodule
        self.label_mapper = self.get_label_map()
        self.transforms = self.get_transforms()
        self.data_module = self.get_datamodule(self.label_mapper, *self.transforms, self.config.splitter)

        # Split Data
        self.data_splits = self.setup_data_splits()

        # Get Dataloaders
        self.get_dataloaders(self.data_module, self.data_splits)

        # Model
        self.model = self.setup_model()

        # Trainer
        self.trainer = self.setup_trainer()
        
    def _split_fn(self):
        fn = self.splitters[self.config.splitter]
        if self.config.label_csv_path:
            label_df = pd.read_csv(self.config.label_csv_path)
            if self.config.splitters == "stratified_group_kfold":
                return partial(fn, label_df=label_df, label_col=self.config.label_col)
        else:
            return fn

            


    def get_datamodule(self, label_mapper, x_transforms, y_transforms, splitter):

        dm = GazeDataModule(self.config.data_folderpath, label_mapper=label_mapper, transform_x=x_transforms, transform_y=y_transforms, batch_size=self.config.batch_size)
        return dm

    def get_dataloaders(self, dm, data_splits):
        # train_dls = []
        # val_dls = []
        if len(data_splits) >= 2:
            train_split, val_split = data_splits[0], data_splits[1]
            train_sampler = SubsetRandomSampler(train_split)
            val_sampler = SubsetRandomSampler(val_split)
            train_dl = dm.train_dataloader(sampler=train_sampler)
            val_dl = dm.val_dataloader(sampler=val_sampler)
            # train_dls.append(train_dl)
            # val_dls.append(val_dl)
        else:
            train_dl = dm.train_dataloader(shuffle=True)
            val_dl = dm.val_dataloader(shuffle=True)
            # train_dls.append(train_dl)
            # val_dls.append(val_dl)
        
        if len(data_splits[2]) > 0:
            test_sampler = SubsetRandomSampler(data_splits[2])
            test_dl = dm.test_dataloader(sampler=test_sampler)
        else:
            test_dl = dm.test_dataloader()
        
        return train_dl, val_dl, test_dl
    
    def setup_datamodule(self, stages=["fit"]):
        for stage in stages:
            self.data_module.setup(stage)

    def _split_fn(self):
        raise NotImplementedError

    def setup_data_splits(self):
        '''
        Split data:
        - cross_validation
            - Stratified Groups
            - Random
        - train, val, test
        '''
        split_fn = self._split_fn()
        train_splits, val_splits, test_splits = split_fn(self.data_module)
        return train_splits, val_splits, test_splits

    def setup_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def setup_trainer(self):
        logger = self.get_logger()
        callbacks = self.get_callbacks()
        trainer = Trainer.from_argparse_args(self.config, logger=logger, callbacks=callbacks)  
        return trainer

    def get_callbacks(self, default_callbacks=[]):
        return default_callbacks

    def get_logger(self, logger_cls=TensorBoardLogger, sub_dir=""):
        if sub_dir:
            logger = logger_cls(self.config.log_dirpath, name=self.config.experiment_logging_folder, sub_dir=sub_dir)
        else:
            logger = logger_cls(self.config.log_dirpath, name=self.config.experiment_logging_folder)
        return logger
    
    def get_label_map(self):
        raise NotImplementedError
    
    def get_transforms(self):
        raise NotImplementedError
    
    def run_cross_val(self):
        metrics_last_epoch = []
        try:
            for train_dl, val_dl, model, trainer in zip(self.train_dls, self.val_dls, self.models, self.trainers):
                metrics_last_epoch.append(self.run_fit(trainer, model, train_dl, val_dl))
        except ValueError as e:
            raise e
        return metrics_last_epoch
        
    def run_fit(self, trainer, model, train_dl, val_dl):
        trainer.fit(model, train_dl, val_dl)
        return trainer.logged_metrics

    def run_test(self, trainer, model, test_dl):
        trainer.test(model, test_dl)    

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Program Args
    group = parser.add_argument_group("Program")
    group.add_argument("--random_seed", type=int, default=42)
    group.add_argument("--config_path", type=str, default="")
    # parser.add_argument("--data_folderpath", type=str)
    # parser.add_argument("--pretrained_weights_dirpath", type=str, default="")
    # parser.add_argument("--decoder_weights_filename", type=str)
    # parser.add_argument("--log_dirpath", type=str, default=".")
    # parser.add_argument("--experiment_logging_folder", type=str, default="")
    # parser.add_argument("--random_seed", type=int, default=42)
    # parser.add_argument("--sequence_length", type=int, nargs="+")
    # parser.add_argument("--stage", type=str, default="fit")
    # parser.add_argument("--use_conv", type=bool, default=True)
    # parser.add_argument("--gru_hidden_size", nargs="+", type=int)

    # ------- Add Arguments ------------ #
    # add training arguments
    parser = Trainer.add_argparse_args(parser)

    # add model arguments
    parser = EncoderClassifierModel.add_model_specific_args(parser)

    # add datamodule arguments
    parser = SequenceToLabelDataModule.add_datamodule_specific_args(parser)
    
    # Parse args
    args = parser.parse_args()

    # if there is a config path then load the arguments from that file
    # if args.config_path:
    #     with open(Path(args.config_path).resolve(), 'r') as f:
    #         args.__dict__ = json.load(f)
    # # no config path, so save the args to a file
    # else:
    #     log_dirpath = Path(args.log_dirpath, args.experiment_logging_folder).resolve()
    #     log_dirpath.mkdir(parents=True, exist_ok=True)
    #     with open(Path(log_dirpath, "command_args.txt").resolve(), 'w') as f:
    #         json.dump(args.__dict__, f, indent=2)

    # ------- Instantiate -------------- #

    datamodule = SequenceToLabelDataModule.from_argparse_args(args)
    model = EncoderClassifierModel.from_argparse_args(args)
    trainer = Trainer.from_argparse_args(args)

    # ----------- Run -------------- #
    trainer.fit(model, datamodule=datamodule)
    