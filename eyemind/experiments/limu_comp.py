
#%%
import yaml
import os
from pytorch_lightning import Trainer, seed_everything
import eyemind
from eyemind.dataloading.gaze_data import VariableLengthSequenceToLabelDataModule
from eyemind.dataloading.limu_bert_loader import GazeformerEmbeddingDataset, gazeformer_embedding_collate_fn, EmbeddingDataModule
from eyemind.models.classifier import ClassifierHead
from pytorch_lightning.cli import LightningCLI

cli = LightningCLI(ClassifierHead, 
                        EmbeddingDataModule, 
                        run=False, 
                        save_config_overwrite=True,
                        )
cli.datamodule.setup()
cli.trainer.fit(cli.model, datamodule=cli.datamodule)