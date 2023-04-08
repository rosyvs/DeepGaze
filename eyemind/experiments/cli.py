from typing import Any
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from .eyemind.trainer.loops import KFoldLoop


class GazeLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--num_folds", type=int, default=1)
        # parser.set_defaults({
        #     "trainer.logger": {
        #         "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
        #         "init_args": {
        #             "save_dir": ""
        #         }
        #     }
        # })

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        trainer = super().instantiate_trainer(**kwargs)
        num_folds = self._get(self.config, "num_folds")
        #export_path = self._get(self.config, "export_path")
        if self.config["subcommand"] == "fit":
            default_fit_loop = trainer.fit_loop
            trainer.fit_loop = KFoldLoop(num_folds)
            trainer.fit_loop.connect(default_fit_loop)
        return trainer
    
        
class FoldsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--fold_number", type=int, default=-1)
        parser.add_argument("--split_filepath", type=str, default="")
        parser.add_argument("--num_folds", type=int, default=4)