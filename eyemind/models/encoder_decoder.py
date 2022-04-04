from argparse import ArgumentParser
from typing import Any
from numpy import vstack
from pytorch_lightning import LightningModule
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch

import sys
from pathlib import Path
sys.path.append(str(Path("../../OBF").resolve()))

from obf.model import ae
from obf.model import creator



class EncoderDecoderModel(LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, criterion: nn.Module, num_classes: int, learning_rate=1e-3, lr_scheduler_step_size=1, freeze_encoder=False, cuda=True):
        super().__init__()
        # Saves hyperparameters (init args)
        self.save_hyperparameters(ignore=["encoder",'decoder'])
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.num_classes = num_classes
        self.auroc_metric = torchmetrics.AUROC(num_classes=num_classes, average="weighted")
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=num_classes)

    def forward(self, X):
        embeddings = self.encoder(X)
        logits = self.decoder(embeddings)
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
        logits = self(X).squeeze()
        logits = logits.reshape(-1, 2)
        y = y.reshape(-1).long()
        logits, y = self._apply_mask(logits, y)
        loss = self.criterion(logits, y)
        preds = self._get_preds(logits)
        probs = self._get_probs(logits)
        y = y.int()
        accuracy = self.accuracy_metric(preds, y)
        auroc = self.auroc_metric(probs, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}_loss": loss})        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        params = self.decoder.parameters() if self.hparams.freeze_encoder else list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_scheduler_step_size, gamma=0.5)}
        return res

    def _apply_mask(self, logits, targets, mask_flag=-1):
        masked_indices = targets == -1
        masked_logits = logits[~masked_indices]
        masked_targets = targets[~masked_indices]
        return masked_logits, masked_targets

    # def _compute_metrics_with_mask(self, probs, y):
    #     masked_indices = y == -1
    #     masked_probs = probs[~masked_indices]
    #     masked_y = y[~masked_indices]
    #     accuracy = self.accuracy_metric(masked_probs, masked_y)
    #     auroc = self.auroc_metric(masked_probs, masked_y)
    #     return accuracy, auroc

    def _get_probs(self, logits):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
        elif self.num_classes > 1:
            probs = torch.softmax(logits, dim=1)        
        return probs

    def _get_preds(self, logits):
        if self.num_classes == 1:
            preds = torch.sigmoid(logits)
        elif self.num_classes > 1:
            preds = torch.argmax(logits, dim=1)
        return preds
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("EncoderDecoderModel")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

