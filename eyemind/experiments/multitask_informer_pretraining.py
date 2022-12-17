from pytorch_lightning.utilities.cli import LightningCLI
from eyemind.models.transformers import InformerEncoderDecoderModel, InformerEncoderFixationModel, InformerMultiTaskEncoderDecoder
from eyemind.dataloading.informer_data import InformerDataModule
from eyemind.experiments.cli import FoldsLightningCLI


if __name__ == "__main__":
    cli = FoldsLightningCLI(InformerMultiTaskEncoderDecoder, InformerDataModule, run=False, seed_everything_default=42)
    cli.datamodule.setup()
    cli.datamodule.load_folds(cli.config.split_filepath)
    cli.datamodule.setup_fold_index(cli.config.fold_number)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
