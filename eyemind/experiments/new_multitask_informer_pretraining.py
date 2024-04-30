from pytorch_lightning.cli import LightningCLI
from eyemind.models.transformers import InformerMultiTaskEncoderDecoder
from eyemind.dataloading.gaze_data import SequenceToMultiLabelDataModule
from eyemind.experiments.cli import FoldsLightningCLI
import os

if __name__ == "__main__":
    # print(f'CWD: {os.getcwd()}')
    cli = FoldsLightningCLI(InformerMultiTaskEncoderDecoder, 
                           SequenceToMultiLabelDataModule, 
                            run=False, 
                            save_config_overwrite=True)
    cli.datamodule.setup()
    # try to load folds from file, otherwise setup the folds 
    try:
        cli.datamodule.load_folds(cli.config.split_filepath)
        print(f'loaded existing folds from split_filepath {cli.config.split_filepath}')
    except Exception as e:        
        print(e)
        print(f'unable to load folds from split_filepath {cli.config.split_filepath}')
        cli.datamodule.setup_folds(cli.config.num_folds)
        cli.datamodule.save_folds(cli.config.split_filepath)
        print(f'generated folds and saved them in split_filepath {cli.config.split_filepath}')

    cli.datamodule.setup_fold_index(cli.config.fold_number)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)