from argparse import ArgumentParser
from pytorch_lightning.utilities.cli import LightningCLI
from eyemind.dataloading.gaze_data import BaseGazeDataModule, BaseSequenceToSequenceDataModule
from eyemind.dataloading.informer_data import InformerDataModule
from eyemind.experiments.cli import GazeLightningCLI

from eyemind.models.transformers import InformerEncoderDecoderModel

if __name__ == "__main__":
    parser = ArgumentParser()
    cli = LightningCLI(InformerEncoderDecoderModel, BaseSequenceToSequenceDataModule, seed_everything_default=42, trainer_defaults={'max_epochs': 5})
