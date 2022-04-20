from pytorch_lightning import Trainer
import torch
from pathlib import Path
from torch.utils.data import SubsetRandomSampler

from eyemind.dataloading.gaze_data import GazeDataModule
from eyemind.dataloading.load_dataset import get_datamodule, get_filenames_for_dataset, get_label_df, get_label_mapper, get_stratified_group_splits, limit_sequence_len
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

    def __init__(self, data_path, label_path, label_col, sequence_len=3000):
        self.label_df = get_label_df(label_path)
        self.label_col = label_col
        self.sequence_len = sequence_len
        self.datamodule = self._get_datamodule()

    @classmethod
    def setup_from_config(cls, experiment_config):
        pass

    def _get_datamodule(self):
        filenames = get_filenames_for_dataset(self.label_df, self.data_path, self.label_col)
        label_mapper = get_label_mapper(self.label_df, self.label_col)
        dm = GazeDataModule(self.data_path, file_list=filenames, label_mapper=label_mapper)

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
    
