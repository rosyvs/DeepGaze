from pytorch_lightning.utilities.cli import LightningCLI
from .eyemind.models.transformers import InformerClassifierModel, InformerEncoderDecoderModel, InformerEncoderFixationModel
from .eyemind.dataloading.gaze_data import SequenceToLabelDataModule


if __name__ == "__main__":
    cli = LightningCLI(InformerClassifierModel, SequenceToLabelDataModule, seed_everything_default=42)
