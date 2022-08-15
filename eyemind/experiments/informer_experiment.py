from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from eyemind.dataloading.gaze_data import BaseGazeDataModule, BaseSequenceToSequenceDataModule, SequenceToLabelDataModule
from eyemind.dataloading.informer_data import InformerDataModule
from eyemind.experiments.cli import GazeLightningCLI
from eyemind.models.encoder_decoder import EncoderDecoderModel, MultiTaskEncoderDecoder, VariableSequenceLengthEncoderDecoderModel

from eyemind.models.transformers import InformerEncoderDecoderModel

model_name_map = {"MultiTaskEncoderDecoder": MultiTaskEncoderDecoder, 
                "InformerEncoderDecoderModel": InformerEncoderDecoderModel, 
                "EncoderDecoderModel": EncoderDecoderModel, 
                "VariableSequenceLengthEncoderDecoderModel": VariableSequenceLengthEncoderDecoderModel}

datamodule_name_map = {"BaseSequenceToSequenceDataModule": BaseSequenceToSequenceDataModule,
                        "InformerDataModule": InformerDataModule,
                        "SequenceToLabelDataModule": SequenceToLabelDataModule}       

def main(args):
    dict_args = vars(args)
    # Setup data
    datamodule_cls = dict_args["datamodule_name"]
    datamodule = datamodule_cls(**dict_args)
    # Setup Model
    model_cls = dict_args["model_name"]
    model = model_cls(**dict_args)
    # Setup Trainer
    trainer = Trainer.from_argparse_args(args)
    # Train
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Choose which model to use
    parser.add_argument("--model_name", type=str, default="MultiTaskEncoderDecoder", help="InformerEncoderDecoderModel, MultiTaskEncoderDecoder, EncoderDecoderModel, VariableSequenceLengthEncoderDecoderModel")
    
    # Choose which datamodule to use
    parser.add_argument("--datamodule_name", type=str, default="BaseSequenceToSequenceDataModule", help="BaseSequenceToSequenceDataModule, InformerDataModule, SequenceToLabelDataModule")
    
    temp_args, _ = parser.parse_known_args()
    model_name_map[temp_args.model_name].add_model_specific_args(parser)
    datamodule_name_map[temp_args.datamodule_name].add_datamodule_specific_args(parser)
    args = parser.parse_args()
    main(args)
    #cli = LightningCLI(model, datamodule, seed_everything_default=42)