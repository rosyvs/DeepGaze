from pathlib import Path
from typing import Any, List, Optional
from pytorch_lightning import LightningModule
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch

from .eyemind.obf.model import ae
from .eyemind.obf.model import creator

def create_encoder(hidden_dim=128, backbone_type='gru', nlayers=2, conv_dim=32,input_dim=2,use_conv=True):
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
    return encoder

class EncoderClassifierModel(LightningModule):
    def __init__(self, encoder_hidden_dim: int=128, encoder_weights_path: str="", classifier_hidden_layers: List[int]=[256,512], n_output: int=1, pos_weight: Optional[List[float]]=None, learning_rate: float=1e-3, dropout: float=0.5, freeze_encoder: bool=False):
        super().__init__()
        # Saves hyperparameters (init args)
        self.save_hyperparameters()
        self.encoder = create_encoder(encoder_hidden_dim)
        if encoder_weights_path:
           self.encoder.load_state_dict(torch.load(encoder_weights_path))
        self.model = creator.create_classifier_from_encoder(self.encoder,hidden_layers=classifier_hidden_layers,n_output=1,dropout=0.5)
        assert(n_output >= 1)
        self.num_classes = n_output
        self.pos_weight = torch.tensor(pos_weight,dtype=float) if pos_weight else None
        if self.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(pos_weight=self.pos_weight)
       
        self.auroc_metric = torchmetrics.AUROC(num_classes=n_output, average="weighted")
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=n_output)

    def forward(self, X):
        logits = self.model(X)
        return logits

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")
    
    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="val")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X, _ = batch
        logits = self(X)
        return self._get_preds(logits) 
    
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
        probs = self._get_probs(logits)
        y = y.int()
        accuracy = self.accuracy_metric(probs, y)
        auroc = self.auroc_metric(probs, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}_loss": loss}, self.current_epoch)        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        params = self.model[1:].parameters() if self.hparams.freeze_encoder else self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_preds(self, logits, threshold=0.5):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        elif self.num_classes > 1:
            preds = torch.argmax(logits, dim=1)
        return preds

    def _get_probs(self, logits):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
        elif self.num_classes > 1:
            probs = torch.softmax(logits, dim=1)
        return probs
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("EncoderClassifierModel")
        group.add_argument('--learning_rate', type=float, default=0.001)
        group.add_argument('--use_conv', type=bool, default=True)
        group.add_argument('--encoder_hidden_dim', type=int, default=128)
        group.add_argument('--class_weights', type=int, nargs='*')
        group.add_argument('--num_classes', type=int, default=2)
        group.add_argument('--freeze_encoder', type=bool, default=False)
        group.add_argument('--encoder_weights_path', type=str, default="")
        group.add_argument('--classifier_hidden_layers', type=int, nargs='+', default=[256, 512])
        return parent_parser

class MeanCombiner(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, X):
        return torch.mean(X,1)

class EncoderClassifierMultiSequenceModel(LightningModule):
    def __init__(self, max_encoder_sequence_length: int=250, encoder_hidden_dim: int=128, encoder_weights_path: str="", classifier_hidden_layers: List[int]=[256,512], pos_weight: Optional[List[float]]=None, n_output: int=1, learning_rate: float=1e-3, dropout: float=0.5, freeze_encoder: bool=False):
        super().__init__()
        # Saves hyperparameters (init args)
        self.save_hyperparameters()
        self.encoder = create_encoder(encoder_hidden_dim)
        if encoder_weights_path:
            self.encoder.load_state_dict(torch.load(encoder_weights_path))
        self.model = creator.create_classifier_from_encoder(self.encoder,hidden_layers=classifier_hidden_layers,n_output=1,dropout=0.5)
        self.combiner = MeanCombiner()
        assert(n_output >= 1)
        self.num_classes = n_output
        self.pos_weight = torch.tensor(pos_weight,dtype=float) if pos_weight else None
        if self.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(pos_weight=self.pos_weight)
       
        self.auroc_metric = torchmetrics.AUROC(num_classes=n_output, average="weighted")
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=n_output)

    def forward(self, X):
        # splits and removes last sequence which might be different shape
        # Should we check if the split part is all zeros?
        X_splits = torch.split(X, self.hparams.max_encoder_sequence_length, dim=1)[:-1]
        X_stacked = torch.stack(X_splits,1)
        batch_size, splits, seq_len, num_features = X_stacked.shape
        X = X_stacked.view(batch_size*splits, seq_len, num_features)
        logits = self.model(X)
        expanded_logits = logits.view(batch_size, splits, self.hparams.n_output)
        outputs = self.combiner(expanded_logits)
        return outputs

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")
    
    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="val")
    
    def predict_step(self, batch, batch_idx):
        X, _ = batch
        logits = self.forward(X).squeeze()
        preds = self._get_preds(logits)
        return preds

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="test")

    def _step(self, batch, batch_idx, step_type):
        try:
            X, y = batch
        except ValueError as e:
            print(f"{batch}")
            raise e
        logits = self(X).squeeze()
        loss = self.criterion(logits, y)
        probs = self._get_probs(logits)
        y = y.int()
        accuracy = self.accuracy_metric(probs, y)
        auroc = self.auroc_metric(probs, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}_loss": loss}, self.global_step)        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        params = self.model[1:].parameters() if self.hparams.freeze_encoder else self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_preds(self, logits, threshold=0.5):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits) 
            preds = (probs > threshold).float()
        else:
            preds = torch.argmax(logits, dim=1)
        return preds

    def _get_probs(self, logits):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
        elif self.num_classes > 1:
            probs = torch.argmax(logits, dim=1)
        return probs
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("EncoderClassifierMultiSequenceModel")
        group.add_argument('--max_encoder_sequence_length', type=int, default=250)
        group.add_argument('--learning_rate', type=float, default=0.001)
        group.add_argument('--use_conv', type=bool, default=True)
        group.add_argument('--encoder_hidden_dim', type=int, default=128)
        group.add_argument('--pos_weight', type=int, nargs='*', default=None)
        group.add_argument('--num_classes', type=int, default=2)
        group.add_argument('--freeze_encoder', type=bool, default=False)
        group.add_argument('--encoder_weights_path', type=str, default="")
        group.add_argument('--classifier_hidden_layers', type=int, nargs='+', default=[256, 512])
        return parent_parser    