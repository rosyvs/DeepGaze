
from eyemind.dataloading.limu_bert_loader import  EmbeddingDataModule
from eyemind.models.classifier import ClassifierHead
from pytorch_lightning.cli import LightningCLI

cli = LightningCLI(
                ClassifierHead, 
                EmbeddingDataModule, 
                run=False, 
                save_config_overwrite=True,
                        )
cli.datamodule.setup()
cli.trainer.fit(cli.model, datamodule=cli.datamodule)