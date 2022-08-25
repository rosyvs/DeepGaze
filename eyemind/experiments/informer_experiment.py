from pytorch_lightning.utilities.cli import LightningCLI
from eyemind.models.transformers import InformerEncoderDecoderModel
from eyemind.dataloading.informer_data import InformerDataModule


if __name__ == "__main__":
    cli = LightningCLI(InformerEncoderDecoderModel, InformerDataModule, seed_everything_default=42)
