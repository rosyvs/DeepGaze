from pytorch_lightning.utilities.cli import LightningCLI
from .eyemind.models.transformers import InformerEncoderDecoderModel, InformerEncoderFixationModel, InformerMultiTaskEncoderDecoder
from .eyemind.dataloading.informer_data import InformerDataModule


if __name__ == "__main__":
    cli = LightningCLI(InformerMultiTaskEncoderDecoder, InformerDataModule, seed_everything_default=42)
