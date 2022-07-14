from functools import reduce
from json import encoder
from pathlib import Path
from turtle import forward
from typing import Any, List
from urllib.request import ProxyBasicAuthHandler
from pytorch_lightning import LightningModule
from sklearn.linear_model import LogisticRegression
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch
from eyemind.dataloading.informer_data import contrastive_batch, predictive_coding_batch, reconstruction_batch
from eyemind.models.loss import RMSELoss

from eyemind.obf.model import ae
from eyemind.obf.model import creator

def load_encoder_decoder(pretrained_weights_dirpath, decoder_weights_filename):
    encoder = creator.load_encoder(str(Path(pretrained_weights_dirpath).resolve()))
    decoder = torch.load(str(Path(pretrained_weights_dirpath, decoder_weights_filename).resolve()),map_location=torch.device('cpu'))
    return encoder, decoder

def create_decoder(hidden_dim=128, out_dim=2, output_seq_length=500, backbone_type="gru", nlayers=2):
    return  ae.RNNDecoder(input_dim=hidden_dim,
                            latent_dim=hidden_dim,
                            out_dim=out_dim,
                            seq_length=output_seq_length,
                            backbone=backbone_type,
                            nlayers=nlayers,
                            batch_norm=True)
                

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
    def __init__(self, sequence_length: int=250, hidden_dim: int=128, class_weights: List[float]=[3.,1.], num_classes: int=2, use_conv: bool=True, learning_rate: float=1e-3, freeze_encoder: bool=False):
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
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,int(self.trainer.max_epochs / 5)), gamma=0.5)}
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

    def _get_preds(self, logits, threshold=0.5):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
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
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        return parser


class VariableSequenceLengthEncoderDecoderModel(EncoderDecoderModel):
    def forward(self, X):
        # Performs predictions with sequence length of self.hparams.sequence_length by splitting the sequences
        # X_splits = torch.split(X, self.hparams.sequence_length, dim=1)
        # X_stacked = torch.stack(X_splits,1)
        # batch_size, splits, seq_len, num_features = X_stacked.shape
        # X = X_stacked.view(batch_size*splits, seq_len, num_features)
        embeddings = self.encoder(X)
        logits = self.decoder(embeddings)
        return logits

    def _step(self, batch, batch_idx, step_type):
        try:
            X, y = batch
        except ValueError as e:
            print(f"{batch}")
            raise e
        logits = self(X).squeeze()
        logits = logits.reshape(-1, 2)
        y = y.reshape(-1).long()
        loss = self.criterion(logits, y)
        preds = self._get_preds(logits)
        probs = self._get_probs(logits)
        y = y.int()
        accuracy = self.accuracy_metric(probs, y)
        auroc = self.auroc_metric(probs, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}": loss}, self.global_step)        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VariableSequenceLengthEncoderDecoderModel")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--sequence_length', type=int, default=250)
        parser.add_argument('--use_conv', type=bool, default=True)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        return parser

class MultiTaskEncoderDecoder(VariableSequenceLengthEncoderDecoderModel):
    def __init__(self, tasks: List[str]=["fi", "pc", "cl", "rc"], sequence_length: int=250, pred_length: int=100, hidden_dim: int=128, class_weights: List[float]=[3.,1.], num_classes: int=2, use_conv: bool=True, learning_rate: float=1e-3, freeze_encoder: bool=False):
        super().__init__(sequence_length, hidden_dim, class_weights, num_classes, use_conv, learning_rate, freeze_encoder)
        self.save_hyperparameters()
        if len(tasks) == 0:
            raise ValueError("There must be at least one task. Length of tasks is 0")
        self.encoder, fi_decoder = create_encoder_decoder(hidden_dim, use_conv, input_seq_length=sequence_length)
        self.decoders = {}
        self.criterions = {}
        self.num_classes = num_classes
        self.metrics = {}
        if "fi" in tasks:
            self.decoders["fi"] = fi_decoder
            self.criterions["fi"] = nn.CrossEntropyLoss(torch.Tensor(class_weights))
            self.metrics["fi"] = torchmetrics.AUROC(num_classes=num_classes, average="weighted")
        if "pc" in tasks:
            self.decoders['pc'] = create_decoder(hidden_dim,output_seq_length=pred_length)
            self.criterions["pc"] = RMSELoss()
            self.metrics["pc"] = torchmetrics.MeanSquaredError()
        if "cl" in tasks:
            cl_input_dim = hidden_dim * 2
            cl_decoder = ae.MLP(input_dim=cl_input_dim,
                                layers=[128, 2],
                                batch_norm=True)
            self.decoders["cl"] = cl_decoder
            self.criterions["cl"] = nn.CrossEntropyLoss()
            self.metrics["cl"] = torchmetrics.Accuracy(num_classes=num_classes)
        if "rc" in tasks:
            self.decoders["rc"] = create_decoder(hidden_dim,output_seq_length=sequence_length)
            self.criterions["rc"] = RMSELoss()
            self.metrics["rc"] = torchmetrics.MeanSquaredError()         

    def forward(self, X, task_name):
        enc = self.encoder(X)
        logits = self.decoders[task_name](enc)
        return logits

    def _step(self, batch, batch_idx, step_type):
        try:
            X, y = batch
        except ValueError as e:
            print(f"{batch}")
            raise e
        total_loss = 0
        for task in self.hparams.tasks:
            if task == "cl":
                X1, X2, y_cl = contrastive_batch(batch[0], self.hparams.sequence_length)
                enc1 = self.encoder(X1)
                enc2 = self.encoder(X2)
                embed = torch.abs(enc1 - enc2)
                logits = self.decoders["cl"](embed).squeeze()
                y_cl = y_cl.reshape(-1).long()
                task_loss = self.criterions[task](logits, y_cl)
                probs = self._get_probs(logits)
                task_metric = self.metrics[task](probs, y_cl.int())
                del X1, X2, y_cl, enc1, enc2, embed, probs
            elif task == "fi":
                logits = self(X, task).squeeze().reshape(-1,2)
                targets_fi = y.reshape(-1).long()
                task_loss = self.criterions[task](logits, targets_fi)
                probs = self._get_probs(logits)
                task_metric = self.metrics[task](probs, targets_fi.int())
                del probs, targets_fi
            elif task == "pc":
                X_pc, y_pc = predictive_coding_batch(batch[0], self.hparams.sequence_length - self.hparams.pred_length, self.hparams.pred_length, 0)
                logits = self(X_pc, task).squeeze()
                assert(logits.shape == y_pc.shape)
                task_loss = self.criterions[task](logits, y_pc)
                task_metric = self.metrics[task](logits, y_pc)
                del X_pc, y_pc
            elif task == "rc":
                logits = self(X, task).squeeze()  
                task_loss = self.criterions[task](logits, X)
                task_metric = self.metrics[task](logits, X)           
            else:
                raise ValueError("Task not recognized.")
            self.log(f"{step_type}_{task}_loss", task_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{step_type}_{task}_metric", task_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            total_loss += task_loss
        return total_loss

    def configure_optimizers(self):
        params = [list(m.parameters()) for m in self.decoders.values()]
        params.append(list(self.encoder.parameters()))
        params = reduce(lambda x,y: x + y, params)
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultiTaskEncoderDecoder")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--sequence_length', type=int, default=250)
        parser.add_argument('--pred_length', type=int, default=100)
        parser.add_argument('--use_conv', type=bool, default=True)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        return parser