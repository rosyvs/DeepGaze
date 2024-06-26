from eyemind.dataloading.gaze_data import SequenceToLabelDataModule
from eyemind.models.classifier import EncoderClassifierModel 
from eyemind.experiments.cli import FoldsLightningCLI


if __name__ == "__main__":
    cli = FoldsLightningCLI(EncoderClassifierModel, 
                            SequenceToLabelDataModule, 
                            run=False, 
                            seed_everything_default=42,
                            save_config_overwrite=True)
    cli.datamodule.setup()
    cli.datamodule.load_folds(cli.config.split_filepath)
    cli.datamodule.setup_fold_index(cli.config.fold_number)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
