from functools import reduce
from pathlib import Path
import random
from typing import List

from pytorch_lightning import LightningModule
import torchmetrics
from torch import nn
import torch.nn.functional as F
import torch
from eyemind.dataloading.batch_loading import contrastive_batch, predictive_coding_batch, reconstruction_batch
from eyemind.models.loss import RMSELoss
from eyemind.dataloading.load_dataset import binarize_labels

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
        self.criterion = nn.CrossEntropyLoss(torch.tensor(class_weights) / torch.tensor(class_weights).sum())
        #self.criterion = nn.CrossEntropyLoss()
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
        if not step_type == 'train': 
            auroc = self.auroc_metric(probs, y) # TODO: put this only at end epoch, possiblr cause of memory leak
            self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)        
        self.logger.experiment.add_scalars("losses", {f"{step_type}": loss}, self.global_step)        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = self.decoder.parameters() if self.hparams.freeze_encoder else list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
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
            preds = torch.argmax(logits, dim=-1)
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
        # TODO: check if there is a way to not do backprop: maybe just return loss = 0 
        logits = self(X).squeeze()
        logits = logits.reshape(-1, 2)
        y = y.reshape(-1).long()
        loss = self.criterion(logits, y)
        #loss = self.fixation_loss(logits, y)
        # if batch_idx == 1:
        #     preds = self._get_preds(logits)
        #     print(preds.sum(), y.sum())
        preds = self._get_preds(logits)
        probs = self._get_probs(logits)
        y = y.int()
        accuracy = self.accuracy_metric(probs, y)
        auroc = self.auroc_metric(probs, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}": loss}, self.global_step)        
        self.log(f"{step_type}_fixations_fraction", {"predicted": preds.float().mean(), "gt": y.float().mean()}, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def fixation_loss(self, logits, targets):
        sac_idx = targets == 0
        sac_loss = self.criterion(logits[sac_idx], targets[sac_idx])
        fix_idx = torch.where(targets == 1)[0]
        fix_idx_rand = fix_idx[torch.randint(len(fix_idx), (len(sac_idx),))]
        fix_loss = self.criterion(logits[fix_idx_rand], targets[fix_idx_rand])
        sac_loss = sac_loss if not torch.isnan(sac_loss) else 0
        fix_loss = fix_loss if not torch.isnan(fix_loss) else 0        
        return sac_loss + fix_loss

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
    def __init__(
        self, 
        tasks: List[str]=["fi", "pc", "cl", "rc"], 
        sequence_length: int=250, 
        pc_seq_length: int=250,
        pred_length: int=100, 
        hidden_dim: int=128, # aka d_model in informer
        class_weights: List[float]=[3.,1.], 
        binarize_threshold: float=0.5,
        max_rmse_err: float=70., 
        num_classes: int=2, 
        use_conv: bool=True, 
        learning_rate: float=1e-3, 
        freeze_encoder: bool=False):
        super().__init__(sequence_length, hidden_dim, class_weights, num_classes, use_conv, learning_rate, freeze_encoder)
        self.save_hyperparameters()
        if len(tasks) == 0:
            raise ValueError("There must be at least one task. Length of tasks is 0")
        self.encoder, fi_decoder = create_encoder_decoder(hidden_dim, out_dim=num_classes, use_conv, input_seq_length=sequence_length)
        #self.decoders = {}
        self.decoders = []
        self.criterions = [] 
        self.num_classes = num_classes
        self.metrics = []
        if "fi" in tasks:
            self.fi_decoder = fi_decoder
            self.decoders.append(fi_decoder)
            self.fi_criterion = nn.CrossEntropyLoss(torch.Tensor(class_weights))
            self.fi_metric = torchmetrics.AveragePrecision(task="multiclass",num_classes=2)
        if "fm" in tasks:
            if "fi" in tasks:
                raise Exception("You can only use one of 'fi' and 'fm' as pretraining tasks, both were provided")
            self.fm_decoder = fi_decoder
            decoders.append(self.fm_decoder)
            self.fm_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
            self.fm_metric = torchmetrics.AveragePrecision(num_classes=num_classes, average="macro", thresholds=20)
        if "pc" in tasks:
            self.pc_decoder = create_decoder(hidden_dim,output_seq_length=pred_length)
            self.decoders.append(self.pc_decoder)
            self.pc_criterion = RMSELoss()
            self.pc_metric = torchmetrics.MeanSquaredError(squared=False)
        if "cl" in tasks:
            cl_input_dim = hidden_dim * 2
            cl_decoder = ae.MLP(input_dim=cl_input_dim,
                                layers=[128, 2],
                                batch_norm=True)
            self.cl_decoder = cl_decoder
            self.decoders.append(self.cl_decoder)
            self.cl_criterion = nn.CrossEntropyLoss()
            self.cl_metric = torchmetrics.Accuracy(num_classes=2)
        if "rc" in tasks:
            self.rc_decoder = create_decoder(hidden_dim,output_seq_length=2) # TODO: why length 2?
            self.decoders.append(self.rc_decoder)
            self.rc_criterion = RMSELoss()
            self.rc_metric = torchmetrics.MeanSquaredError(squared=False)     
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, X, task_name):
        enc = self.encoder(X)
        logits = self.decoders[task_name](enc)
        return logits

    # def predict_task(self, batch, task):
    #     X, fix_y, X2, cl_y = batch
    #     if task == "cl":
    #         enc1 = self.encoder(X)
    #         enc2 = self.encoder(X2)
    #         embed = torch.abs(enc1 - enc2)
    #         logits = self.cl_decoder(embed).squeeze()
    #         preds = self._get_preds(logits)
    #         targets = cl_y
    #     elif task == "fi":
    #         enc = self.encoder(X)
    #         logits = self.fi_decoder(enc).squeeze()
    #         preds = self._get_preds(logits)
    #         targets = binarize_labels(fix_y.reshape(-1).long(), self.binarize_threshold) # ensures labels are binary even if mroe classes in file. 
    #     elif task == "fm":
    #         enc = self.encoder(X)
    #         logits = self.fm_decoder(enc).squeeze()
    #         preds = self._get_preds(logits)
    #         targets = fix_y
    #     elif task == "pc":
    #         X_pc, Y_pc = self.predictive_coding_batch(batch[0])
    #         enc = self.encoder(X_pc)
    #         logits = self.pc_decoder(enc).squeeze()
    #         targets = Y_pc
    #     elif task == "rc":
    #         enc = self.encoder(X)
    #         logits = self.rc_decoder(enc).squeeze()
    #         targets = X
    #     else:
    #         raise ValueError("Task not recognized.")        
    #     return preds, targets

    def predict_step(self, batch, batch_idx):
        X, fix_y, X2, cl_y = batch
        task_predictions = {}
        for task in self.hparams.tasks:
            if task == "cl":
                enc1 = self.encoder(X)
                enc2 = self.encoder(X2)
                embed = torch.abs(enc1 - enc2)
                logits = self.cl_decoder(embed).squeeze()
                preds = self._get_preds(logits)
                task_predictions["cl"] = preds
                del X2, enc1, enc2, embed, preds
            elif task == "fi":
                enc = self.encoder(X)
                logits = self.fi_decoder(enc).squeeze()
                preds = self._get_preds(logits)
                task_predictions["fi"] = preds
                del enc
            elif task == "fm":
                enc = self.encoder(X)
                logits = self.fi_decoder(enc).squeeze()
                preds = self._get_preds(logits)
                task_predictions["fm"] = preds
                del enc
            elif task == "pc":
                X_pc, Y_pc = predictive_coding_batch(X, self.hparams.pc_seq_length, self.hparams.pred_length, self.hparams.label_length)
                enc = self.encoder(X_pc)
                logits = self.pc_decoder(enc).squeeze()
                assert(logits.shape == Y_pc.shape)
                task_predictions["pc"] = logits
                del X_pc, Y_pc
            elif task == "rc":
                enc = self.encoder(X)
                logits = self.rc_decoder(enc).squeeze()
                task_predictions["rc"] = logits    
            else:
                raise ValueError("Task not recognized.")
        return task_predictions
               
    def _step(self, batch, batch_idx, step_type):
        try:
            X, fix_y, X2, cl_y = batch
        except ValueError as e:
            print(f"{batch}")
            raise e
        total_loss = 0
        for task in self.hparams.tasks:
            if task == "cl":
                enc1 = self.encoder(X)
                enc2 = self.encoder(X2)
                embed = torch.abs(enc1 - enc2)
                logits = self.cl_decoder(embed).squeeze()
                cl_y = cl_y.reshape(-1).long()
                task_loss = self.cl_criterion(logits, cl_y)
                probs = self._get_probs(logits)
                task_metric = self.cl_metric(probs, cl_y.int())
                del X2, cl_y, enc1, enc2, embed, probs
            elif task == "fi":
                enc = self.encoder(X)
                logits = self.fi_decoder(enc).squeeze().reshape(-1,2)
                targets_fi = binarize_labels(fix_y.reshape(-1).long(), self.binarize_threshold) # ensures labels are binary even if mroe classes in file. 
                task_loss = self.fi_criterion(logits, targets_fi)
                probs = self._get_probs(logits)
                task_metric = self.fi_metric(probs, targets_fi.int())
                del enc, probs, targets_fi
            elif task == "fm":
                enc = self.encoder(X)
                logits = self.fm_decoder(enc).squeeze().reshape(-1,self.c_out) #TODO: reshape this differently or not at all? 
                targets_fm = fix_y.reshape(-1).long() # remove reshape #TODO: reshape to be long, but unreshape the logits
                task_loss = self.fm_criterion(logits, targets_fm)
                probs = self._get_probs(logits)
                task_metric = self.fm_metric(probs, targets_fm)
                del enc, probs, targets_fm, fix_y
            elif task == "pc":
                X_pc, Y_pc = predictive_coding_batch(X, self.hparams.pc_seq_length, self.hparams.pred_length, self.hparams.label_length)
                enc = self.encoder(X_pc)
                logits = self.pc_decoder(enc).squeeze()
                assert(logits.shape == Y_pc.shape) #
                mask = Y_pc > -180
                task_loss = self.pc_criterion(logits[mask], Y_pc[mask])
                task_metric = self.pc_metric(logits[mask], Y_pc[mask])
                #task_loss = torch.clamp(self.pc_criterion(logits, Y_pc), max=self.hparams.max_rmse_err)
                #task_metric = self.pc_metric(logits, Y_pc)
                del X_pc, Y_pc
            elif task == "rc":
                enc = self.encoder(X)
                logits = self.rc_decoder(enc).squeeze()
                mask = X > -180
                task_loss = self.rc_criterion(logits[mask], X[mask])
                task_metric = self.rc_metric(logits[mask], X[mask])
                #task_loss = torch.clamp(self.rc_criterion(logits, X), max=self.hparams.max_rmse_err)
                #task_metric = self.rc_metric(logits, X)               
            else:
                raise ValueError("Task not recognized.")
            self.log(f"{step_type}_{task}_loss", task_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"{step_type}_{task}_metric", task_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            total_loss += task_loss
        self.log(f"{step_type}_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        params = [list(m.parameters()) for m in self.decoders]
        params.append(list(self.encoder.parameters()))
        params = reduce(lambda x,y: x + y, params)
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res    



    # def contrastive_batch(self, X_batch, cl_y):
    #     n,_,fs = X_batch.shape
    #     cl_seq_len=self.hparams.sequence_length//2
    #     x1 = torch.zeros((n, cl_seq_len, fs), device=self.device)
    #     x2 = torch.zeros((n, cl_seq_len, fs), device=self.device)
    #     # Random 0 (different scanpath) or 1 (same scanpath)
    #     y = torch.randint(low=0, high=2,size=(n,))
    #     for i in range(n):
    #         # Pick sequence from different cl_y label
    #         start_ind = random.randrange(0,cl_seq_len)
    #         if y[i] == 0:
    #             # Select sequences with different label
    #             X_dif_label = X_batch[cl_y != cl_y[i]]
    #             # Set the contrastive sequence to be one of those labels with different sequence
    #             x2[i] = X_dif_label[random.randrange(0, X_dif_label.shape[0]), start_ind:start_ind+cl_seq_len]
    #         # Choose from same scanpath
    #         elif y[i] == 1:
    #             # Get indices of X where subsequences in same scanpath
    #             indices = torch.arange(n)[cl_y == cl_y[i]]
    #             # Exclude own index
    #             #indices = indices[indices!=i]
    #             x2[i] = X_batch[indices[random.randrange(0, indices.shape[0])],start_ind:start_ind+cl_seq_len]
    #         else:
    #             raise ValueError("y label can only be 0 or 1 for contrastive learning")
    #     return x1, x2, y.float()

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
        parser.add_argument('--max_rmse_err', type=float, default=70., help='clamps max rmse loss')
        return parser