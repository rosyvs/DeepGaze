from argparse import ArgumentParser
import argparse
from cgi import test
import math
from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import yaml
from eyemind.dataloading.gaze_data import BaseSequenceToSequenceDataModule, SequenceToSequenceDataModule
from eyemind.models.encoder_decoder import VariableSequenceLengthEncoderDecoderModel

name_to_cls = {"trainer": Trainer, "model": VariableSequenceLengthEncoderDecoderModel, "data": SequenceToSequenceDataModule}

def init_trainer(config, logger=None, callbacks=None):
    Trainer()

def instantiate_class_from_config(config, cls_name, logger=None):
    args = config[cls_name]
    if logger and cls_name == "trainer":
        return name_to_cls[cls_name](**args, logger=logger)
    return name_to_cls[cls_name](**args)

def instantiate_lightningmodules(config):
    modules = [instantiate_class_from_config(config, k) for k in name_to_cls.keys()]
    return tuple(modules)

def combine_hyperparameter_config(config, hp_config):
    for hp_k, hp_v in hp_config.items():
        for _, l_module_dict in config.items():
            if isinstance(l_module_dict, dict):
                if hp_k in l_module_dict:
                    l_module_dict[hp_k] = hp_v
    return config

def train_tune(hyperparameter_config, lightning_config, num_gpus=0):
    config = combine_hyperparameter_config(lightning_config, hyperparameter_config)
    config["trainer"]["gpus"] = math.ceil(num_gpus)
    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")
    model = VariableSequenceLengthEncoderDecoderModel(**config["model"])
    datamodule = BaseSequenceToSequenceDataModule(**config["data"])
    tunecallback = TuneReportCheckpointCallback(
                    metrics={
                        "val_loss": "val_loss",
                        "val_auroc": "val_auroc"},
                    filename="checkpoint",
                    on="validation_end"
                    )
    config["trainer"]["logger"] = logger
    config["trainer"]["callbacks"] = [tunecallback]
    trainer = Trainer(**config['trainer'])
    trainer.fit(model, datamodule=datamodule)    

def tune_seq_hidden(lightning_config, num_samples=1, gpus_per_trial=0):
    tune_config = {"sequence_length": tune.grid_search([100,250,500]),
        "hidden_dim": tune.grid_search([128, 256])}
    num_epochs = lightning_config['trainer']['max_epochs']
    scheduler = ASHAScheduler(
    max_t=num_epochs,
    grace_period=1,
    reduction_factor=2)

    reporter = CLIReporter(parameter_columns=["sequence_length", "hidden_dim"],
                            metric_columns=["val_loss", "val_auroc", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_tune, 
                                                    lightning_config=lightning_config, 
                                                    num_gpus=gpus_per_trial)

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="val_loss",
        mode="min",
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./ray_results",
        name="tune_fixation")

    print("Best hyperparameters found were: ", analysis.best_config)        

def test_train_tune(lightning_config, num_gpus=0):
    hyperparameter_config = {"sequence_length": 500, "hidden_dim": 256}
    train_tune(hyperparameter_config, lightning_config)

if __name__ == "__main__":
    parser = ArgumentParser("Tunes hyperparameters with RayTune")
    parser.add_argument("-c", type=str, help="lightning configuration path")
    parser.add_argument("--gpus_per_trial", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()
    with open(args.c, 'r') as f:
        lightning_config = yaml.safe_load(f)
    pytorch_lightning.seed_everything(lightning_config["seed_everything"], workers=True)
    tune_seq_hidden(lightning_config, num_samples=args.num_samples, gpus_per_trial=args.gpus_per_trial)
    #test_train_tune(lightning_config)