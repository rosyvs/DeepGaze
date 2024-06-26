from eyemind.dataloading.gaze_data import PIDkFoldS2SDataModule
from eyemind.models.encoder_decoder import MultiTaskEncoderDecoder
from eyemind.experiments.cli import FoldsLightningCLI
import os

if __name__ == "__main__":
    print(f'CWD: {os.getcwd()}')
    cli = FoldsLightningCLI(MultiTaskEncoderDecoder, 
                            PIDkFoldS2SDataModule,
                            run=False, 
                            seed_everything_default=42, 
                            save_config_overwrite=True)
    cli.datamodule.setup()
    if cli.config.num_folds != -1:
        cli.datamodule.setup_folds(cli.config.num_folds)
        cli.datamodule.save_folds(cli.config.split_filepath)
    else:
        cli.datamodule.load_folds(cli.config.split_filepath)
    cli.datamodule.setup_fold_index(cli.config.fold_number)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
