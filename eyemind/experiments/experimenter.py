from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning
import torch
from pathlib import Path
from torch.utils.data import SubsetRandomSampler

from eyemind.dataloading.gaze_data import GazeDataModule
from eyemind.dataloading.load_dataset import get_datamodule, get_filenames_for_dataset, get_label_df, get_label_mapper, get_stratified_group_splits, limit_sequence_len
#from eyemind.experiments.fixation_experiment import setup_data_module
from eyemind.models.classifier import EncoderClassifierModel
from eyemind.obf.model import ae
from eyemind.obf.model import creator

class BaseExperiment():

    def __init__(self, data_module, models, trainers, train_val_splits=None, test_split=None):
        self.data_module = data_module
        self.models = models
        self.trainers = trainers
        self.train_val_splits = train_val_splits
        self.test_split = test_split
        self.train_dls, self.val_dls, self.test_dls = self.get_dataloaders()

    @classmethod
    def setup_from_config(cls, experiment_config):
        pass

    def get_dataloaders(self):
        train_dls = []
        val_dls = []
        if self.train_val_splits:
            for train_split, val_split in self.train_val_splits:
                train_sampler = SubsetRandomSampler(train_split)
                val_sampler = SubsetRandomSampler(val_split)
                train_dl = self.data_module.train_dataloader(sampler=train_sampler)
                val_dl = self.data_module.val_dataloader(sampler=val_sampler)
                train_dls.append(train_dl)
                val_dls.append(val_dl)
        else:
            train_dl = self.data_module.train_dataloader(shuffle=True)
            val_dl = self.data_module.val_dataloader(shuffle=True)
            train_dls.append(train_dl)
            val_dls.append(val_dl)
        
        if self.test_split:
            test_sampler = SubsetRandomSampler(self.test_split)
            test_dl = self.data_module.test_dataloader(sampler=test_sampler)
        else:
            test_dl = self.data_module.test_dataloader()
        
        return train_dls, val_dls, test_dl

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


class Experiment():

    def __init__(self, config):
        self.config = config
        self.setup()      

    def setup(self):
        # Seed everything for reproducability
        pytorch_lightning.seed_everything(self.config.random_seed, workers=True)

        # Create datamodule
        self.label_mapper = self.get_label_map()
        self.transforms = self.get_transforms()
        self.data_module = self.get_datamodule(self.label_mapper, *self.transforms)

        # Split Data
        self.data_splits = self.setup_data_splits()

        # Get Dataloaders
        self.get_dataloaders(self.data_module, self.data_splits)

        # Model
        self.model = self.setup_model()

        # Trainer
        self.trainer = self.setup_trainer()

    def get_datamodule(self, label_mapper, x_transforms, y_transforms):
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



def get_encoder_classifier_model(pretrained_weights_dir, model_args):
    encoder = creator.load_encoder(pretrained_weights_dir)
    model = EncoderClassifierModel(encoder, **model_args)
    return model
    
def train(model, train_dl, val_dl, trainer_args):
    trainer = Trainer(**trainer_args)
    trainer.fit(model, train_dl, val_dl)

def train_one_split():
    pass

def train_one_label(data_path, label_path, label_col, split=False):
    label_df = get_label_df(label_path)
    dm = get_datamodule(label_col, label_df, data_path, x_transforms=[limit_sequence_len,lambda data: torch.tensor(data).float()], y_transforms=[lambda data: torch.tensor(data).float()])
    if split:
        files = [f.split(".")[0] for f in dm.file_list]
        splits = get_stratified_group_splits( files, label_df, label_col)
    
