from argparse import ArgumentParser
from pathlib import Path
from typing import Any
from numpy import vstack
from pytorch_lightning import LightningModule
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch

from eyemind.obf.model import ae
from eyemind.obf.model import creator

def load_encoder_decoder(pretrained_weights_dirpath, decoder_weights_filename):
    encoder = creator.load_encoder(str(Path(pretrained_weights_dirpath).resolve()))
    decoder = torch.load(str(Path(pretrained_weights_dirpath, decoder_weights_filename).resolve()),map_location=torch.device('cpu'))
    return encoder, decoder

def create_encoder_decoder(hidden_dim=128, use_conv=True, conv_dim=32, input_dim=2, out_dim=2, input_seq_length=500, backbone_type="gru", nlayers=2):
    if use_conv:
        enc_layers = [
            ae.CNNEncoder(input_dim=input_dim, latent_dim=conv_dim, layers=[
                16,
            ]),
            ae.RNNEncoder(input_dim=conv_dim,
                        latent_dim=hidden_dim,
                        backbone=backbone_type,
                        nlayers=nlayers,
                        layer_norm=False)
        ] 
    else: 
        enc_layers = [ae.RNNEncoder(input_dim=input_dim,
                        latent_dim=hidden_dim,
                        backbone=backbone_type,
                        nlayers=nlayers,
                        layer_norm=False)]

    encoder = nn.Sequential(*enc_layers)

    fi_decoder = ae.RNNDecoder(input_dim=hidden_dim,
                               latent_dim=hidden_dim,
                               out_dim=out_dim,
                               seq_length=input_seq_length,
                               backbone=backbone_type,
                               nlayers=nlayers,
                               batch_norm=True)

    return encoder, fi_decoder    

class EncoderDecoderModel(LightningModule):
    def __init__(self, sequence_length: int, hidden_dim: int, class_weights: list, num_classes: int, use_conv=True, learning_rate=1e-3, lr_scheduler_step_size=1, freeze_encoder=False):
        super().__init__()
        # Saves hyperparameters (init args)
        self.save_hyperparameters()
        self.encoder, self.decoder = create_encoder_decoder(hidden_dim, use_conv, input_seq_length=sequence_length)
        self.criterion = nn.CrossEntropyLoss(torch.Tensor(class_weights))
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
        #print(preds.sum(), y.sum())
        accuracy = self.accuracy_metric(probs, y)
        auroc = self.auroc_metric(probs, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}": loss}, self.global_step)        
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
        masked_indices = targets == mask_flag
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
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--sequence_length', type=int, default=250)
        parser.add_argument('--use_conv', type=bool, default=True)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--class_weights', type=int, nargs='*', default=[3., 1.])
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        return parser

