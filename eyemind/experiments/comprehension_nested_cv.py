from argparse import ArgumentParser
import math
from pathlib import Path
import pandas as pd
from pytorch_lightning import Trainer
import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import yaml
from .eyemind.dataloading.gaze_data import SequenceToLabelDataModule
from .eyemind.models.classifier import EncoderClassifierModel, EncoderClassifierMultiSequenceModel
import ray

name_to_cls = {"trainer": Trainer, "model": EncoderClassifierMultiSequenceModel, "data": SequenceToLabelDataModule}

def init_trainer(config, logger=None, callbacks=None):
    pass

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

def get_dir_exp_name(exp_name, outer_index, inner_index, local_dir="./ray_results"):
    local_dir = str(Path(local_dir, exp_name).resolve())
    fold_exp_name = f"exp_name_outer{outer_index}_inner{inner_index}"
    return local_dir, fold_exp_name

def get_fold_hparam_name(hparam_config, fold_idx):
    s = f"fold{fold_idx}_"
    for k,v in hparam_config.items():
        s += f"{k}={v}"
    return s
    
def get_best_hp_config(analyses, hparams, metric="val_auroc", mode="max"):
    hparam_cols = [f"config/{hparam}" for hparam in hparams]
    res_dfs = [analysis.dataframe(metric, mode) for analysis in analyses]
    full_results = pd.concat(res_dfs).groupby(hparam_cols, as_index=False)[metric].mean()
    idx = full_results[metric].idxmax()
    hparam_df = full_results[hparam_cols].iloc[idx]
    hparam_cols_dict =hparam_df.to_dict()
    res_dict = {}
    for k, v in hparam_cols_dict.items():
        key = k.split("/")[1]
        res_dict[key] = v
    return res_dict

def run_train(hyperparameter_config, lightning_config, datamodule, train_ds, val_ds, logger, num_gpus=0):
    print("STARTING OUTER FOLD TRAINING")
    config = combine_hyperparameter_config(lightning_config, hyperparameter_config)
    config["trainer"]["gpus"] = math.ceil(num_gpus)
    #model = EncoderClassifierMultiSequenceModel(**config["model"])
    model = EncoderClassifierModel(**config["model"])
    config["trainer"]["logger"] = logger
    #config["trainer"]["strategy"] = "ddp"
    trainer = Trainer(**config['trainer'])
    trainer.fit(model, train_dataloaders=datamodule.get_dataloader(train_ds), val_dataloaders=datamodule.get_dataloader(val_ds))
    print("OUTER FOLD TRAINING FINISHED")

def run_tune(hyperparameter_config, lightning_config, datamodule, train_ds, val_ds, num_gpus=0):
    '''
    Fits and evaluates model with given hyperparameters on given train and val datasets.
    '''
    config = combine_hyperparameter_config(lightning_config, hyperparameter_config)
    config["trainer"]["gpus"] = math.ceil(num_gpus)
    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")
    #model = EncoderClassifierMultiSequenceModel(**config["model"])
    model = EncoderClassifierModel(**config["model"])
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
    trainer.fit(model, train_dataloaders=datamodule.get_dataloader(train_ds), val_dataloaders=datamodule.get_dataloader(val_ds))

def nested_cv_tune(lightning_config, num_outer_folds=4, num_inner_folds=4, num_samples=1, gpus_per_trial=0, exp_name="comprehension_tuning"):
    # Tuning configurations and reporting
    tune_config = {"freeze_encoder": tune.grid_search([True, False])}
    hparam_keys = list(tune_config.keys())
    num_epochs = lightning_config['trainer']['max_epochs']
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}
    # Setup logging directories
    local_dir = "ray_results"
    Path(local_dir, exp_name).mkdir(parents=True, exist_ok=True)
    # Datamodule creation and setup folds
    datamodule = SequenceToLabelDataModule(**lightning_config['data'])
    datamodule.setup()
    datamodule.setup_cv_folds(num_outer_folds, num_inner_folds)

    # Running Nested Cross Validation
    # outer loop
    for i in range(num_outer_folds):
        analyses = []
        # inner loop
        for j in range(num_inner_folds):
            train_fold, val_fold = datamodule.get_cv_fold(i, j)
            train_fn_with_parameters = tune.with_parameters(run_tune,
                                                            lightning_config=lightning_config,
                                                            datamodule=datamodule,
                                                            train_ds=train_fold,
                                                            val_ds=val_fold,
                                                            num_gpus=gpus_per_trial)
            fold_local_dir, fold_exp_name = get_dir_exp_name(exp_name, i, j, local_dir)
            scheduler = ASHAScheduler(max_t=num_epochs,
                                        grace_period=num_epochs//2,
                                        reduction_factor=2)
            reporter = CLIReporter(parameter_columns=["freeze_encoder"],
                                    metric_columns=["val_loss", "val_auroc", "training_iteration"])
            analysis = tune.run(train_fn_with_parameters,
                                resources_per_trial=resources_per_trial,
                                metric="val_loss",
                                mode="min",
                                config=tune_config,
                                num_samples=num_samples,
                                scheduler=scheduler,
                                progress_reporter=reporter,
                                local_dir=fold_local_dir,
                                name=fold_exp_name,
                                keep_checkpoints_num=1,
                                checkpoint_score_attr="val_loss",
                                checkpoint_at_end=True
                                )
            analyses.append(analysis)
        
        # Calculate best hyperparameter config by taking the mean of the auroc scores
        hp_config = get_best_hp_config(analyses, hparam_keys)
        print(f"Best hyperparameter config: {hp_config}")
        # Pass best hyperparameter config to run_train
        train_fold, val_fold = datamodule.get_cv_fold(i,-1)
        logger = TensorBoardLogger(save_dir=local_dir, name=exp_name, version=get_fold_hparam_name(hp_config, i))
        print(f"Logging to: {logger.log_dir}")
        run_train(hp_config, lightning_config, datamodule, train_fold, val_fold, logger, num_gpus=min(gpus_per_trial*2, 4))

    

if __name__ == "__main__":
    parser = ArgumentParser("Tunes hyperparameters with RayTune")
    parser.add_argument("-c", type=str, help="lightning configuration path")
    parser.add_argument("--gpus_per_trial", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_outer_folds", type=int, default=4)
    parser.add_argument("--num_inner_folds", type=int, default=4)
    parser.add_argument("--exp_name", type=str, default="comprehension_nestedcv")
    args = parser.parse_args()
    with open(args.c, 'r') as f:
        lightning_config = yaml.safe_load(f)
    pytorch_lightning.seed_everything(lightning_config["seed_everything"], workers=True)
    nested_cv_tune(lightning_config, num_samples=args.num_samples, gpus_per_trial=args.gpus_per_trial, exp_name=args.exp_name, num_outer_folds=args.num_outer_folds, num_inner_folds=args.num_inner_folds)
