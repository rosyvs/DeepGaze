from pytorch_lightning.cli import LightningCLI
from eyemind.dataloading.gaze_data import BaseSequenceToSequenceDataModule

from eyemind.models.encoder_decoder import MultiTaskEncoderDecoder


if __name__ == "__main__":
    cli = LightningCLI(MultiTaskEncoderDecoder, BaseSequenceToSequenceDataModule, seed_everything_default=42)
