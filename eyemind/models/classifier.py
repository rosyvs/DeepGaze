from argparse import ArgumentParser
from numpy import vstack
from pytorch_lightning import LightningModule
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch

from eyemind.obf.model import ae
from eyemind.obf.model import creator


class EncoderClassifierModel(LightningModule):
    def __init__(self, encoder: nn.Module, hidden_layers=[256,512], n_output: int=1, learning_rate=1e-3, lr_scheduler_step_size=1, dropout=0.5, freeze_encoder=False, cuda=True):
        super().__init__()
        # Saves hyperparameters (init args)
        self.save_hyperparameters(ignore="encoder")
        self.model = creator.create_classifier_from_encoder(encoder,hidden_layers=hidden_layers,n_output=1,dropout=0.5)
        assert(n_output >= 1)
        self.num_classes = n_output
        if self.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
       
        self.auroc_metric = torchmetrics.AUROC(num_classes=n_output, average="weighted")
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=n_output)

    def forward(self, X):
        logits = self.model(X)
        return logits

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")
    
    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="val")
    
    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="test")

    def _step(self, batch, batch_idx, step_type):
        try:
            X, y = batch
        except ValueError as e:
            print(f"{batch}")
            raise e
        logits = self.model(X).squeeze()
        loss = self.criterion(logits, y)
        preds = self._get_preds(logits)
        y = y.int()
        accuracy = self.accuracy_metric(preds, y)
        auroc = self.auroc_metric(preds, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}_loss": loss})        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        params = self.model[1:].parameters() if self.hparams.freeze_encoder else self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step_size, gamma=0.5)}
        return res

    def _get_preds(self, logits):
        if self.num_classes == 1:
            preds = torch.sigmoid(logits)
        elif self.num_classes > 1:
            preds = torch.argmax(logits, dim=1)
        return preds
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EncoderClassifierModel")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

