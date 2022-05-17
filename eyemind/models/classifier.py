from typing import List
from pytorch_lightning import LightningModule
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch

from eyemind.obf.model import ae
from eyemind.obf.model import creator

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
    def __init__(self, encoder_hidden_dim: int=128, encoder_weights_path: str="", classifier_hidden_layers: List[int]=[256,512], n_output: int=1, learning_rate: float=1e-3, dropout: float=0.5, freeze_encoder: bool=False):
        super().__init__()
        # Saves hyperparameters (init args)
        self.save_hyperparameters()
        if encoder_weights_path:
            self.encoder = creator.load_encoder(encoder_weights_path)
        else:
            self.encoder = create_encoder(encoder_hidden_dim)
        self.model = creator.create_classifier_from_encoder(self.encoder,hidden_layers=classifier_hidden_layers,n_output=1,dropout=0.5)
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
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_preds(self, logits):
        if self.num_classes == 1:
            preds = torch.sigmoid(logits)
        elif self.num_classes > 1:
            preds = torch.argmax(logits, dim=1)
        return preds
        
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

