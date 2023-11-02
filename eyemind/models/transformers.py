from functools import reduce
from typing import Any, List
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from eyemind.analysis.predictions import get_encoder_from_checkpoint
from eyemind.dataloading.batch_loading import fixation_batch, predictive_coding_batch
from eyemind.obf.model import ae
from ..dataloading.transforms import StandardScaler
from eyemind.models.informer.models.model import InformerStack

from eyemind.models.informer.utils.masking import TriangularCausalMask, ProbMask
from eyemind.models.informer.models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from eyemind.models.informer.models.decoder import Decoder, DecoderLayer
from eyemind.models.informer.models.attn import FullAttention, ProbAttention, AttentionLayer
from eyemind.models.informer.models.embed import GazeEmbedding
from eyemind.models.loss import RMSELoss

class InformerEncoder(nn.Module):
    def __init__(self,
                enc_in: int=2,
                factor: int=5, #ProbSparse sampling factor (only makes affect when attention_type=“prob”). It is used to control the reduced query matrix (Q_reduce) input length.
                d_model: int=512, 
                n_heads: int=8, 
                e_layers: int=3,
                d_ff: int=512, 
                dropout: float=0.05, 
                attn: str='prob', 
                activation: str='gelu', 
                output_attention: bool=False, 
                distil: bool=True, 
                ):
        super().__init__() 
        self.output_attention = output_attention
        # Encoder
        self.enc_embedding = GazeEmbedding(enc_in, d_model, dropout)
        Attn = ProbAttention if attn=='prob' else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out

class InformerDecoder(nn.Module):
    def __init__(self,                 
                dec_in: int=1, 
                c_out: int=2, 
                factor: int=5, 
                d_model: int=512, 
                n_heads: int=8, 
                d_layers: int=2, 
                d_ff: int=512, 
                dropout: float=0.05, 
                attn: str='prob', 
                activation: str='gelu', 
                mix: bool=True,
                ):
        super().__init__()
        Attn = ProbAttention if attn=='prob' else FullAttention

        self.dec_embedding = GazeEmbedding(dec_in, d_model, dropout)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, enc_out, x_dec, dec_self_mask=None, dec_enc_mask=None, pred_len=0):
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        if pred_len:
            return dec_out[:, -pred_len:]
        else:                     
            return dec_out

class InformerEncoderDecoderModel(LightningModule):
    def __init__(self, 
                enc_in: int=2, 
                dec_in: int=1, 
                c_out: int=2, 
                seq_len: int=250, 
                label_len: int=100, 
                pred_len: int=150,
                padding: int=0,
                factor: int=5, 
                d_model: int=512, 
                n_heads: int=8, 
                e_layers: int=3, 
                d_layers: int=2, 
                d_ff: int=512, 
                dropout: float=0.05, 
                attn: str='prob', 
                activation: str='gelu', 
                output_attention: bool=False, 
                distil: bool=True, 
                mix: bool=True, 
                class_weights: List[float]=[3.,1.],
                max_rmse_err: float=70., 
                learning_rate: float=1e-3, 
                freeze_encoder: bool=False):
        super().__init__()
        self.save_hyperparameters()
        # Scaler
        self.scaler = StandardScaler()
        # Loss function
        self.pc_criterion = RMSELoss()
        # Metrics
        self.pc_metric = torchmetrics.MeanSquaredError(squared=False) 
        # Encoding
        self.enc_embedding = GazeEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = GazeEmbedding(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)  
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.hparams.output_attention:
            return dec_out[:,-self.hparams.pred_len:,:], attns
        else:
            return dec_out[:,-self.hparams.pred_len:,:] # [B, L, D]

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")
    
    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="val")
    
    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="test")

    def _step(self, batch, batch_idx, step_type):
        try:
            X, fix_y = batch
        except ValueError as e:
            print(f"{batch}")
            raise e

        # Predictive Coding:
        X_pc, Y_pc = predictive_coding_batch(X, self.hparams.seq_len, self.hparams.pred_len, self.hparams.label_len)
        if self.hparams.output_attention:
            logits = self(X_pc, Y_pc)[0]
        else:
            logits = self(X_pc, Y_pc)
        logits = logits.squeeze()
        assert(logits.shape == Y_pc.shape)
        mask = Y_pc > -180
        #task_loss = torch.clamp(self.pc_criterion(logits, Y_pc),max=self.hparams.max_rmse_err)
        task_loss = self.pc_criterion(logits[mask], Y_pc[mask])
        logits = self.scaler.inverse_transform(logits)
        Y_pc = self.scaler.inverse_transform(Y_pc)
        task_metric = self.pc_metric(logits[mask], Y_pc[mask])
        self.log(f"{step_type}_loss", task_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_pc_metric", task_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return task_loss

    def configure_optimizers(self):
        params = self.decoder.parameters() if self.hparams.freeze_encoder else list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_probs(self, logits):
        if self.hparams.c_out == 1:
            probs = torch.sigmoid(logits)
        elif self.hparams.c_out > 1:
            probs = torch.softmax(logits, dim=1)        
        return probs

    def _get_preds(self, logits, threshold=0.5):
        if self.hparams.c_out == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        elif self.hparams.c_out > 1:
            preds = torch.argmax(logits, dim=1)
        return preds
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InformerEncoderDecoderModel")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--sequence_length', type=int, default=250)
        parser.add_argument('--label_len', type=int, default=100, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=150, help='prediction sequence length')
        parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=2, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        parser.add_argument('--max_rmse_err', type=float, default=70., help='clamps max rmse loss')
        return parser


class InformerEncoderFixationModel(LightningModule):
    def __init__(self, 
                enc_in: int=2, # one neuron for each of X and Y coord
                dec_in: int=1, 
                c_out: int=2, # one neuron for each of X and Y coord
                seq_len: int=250, 
                label_len: int=100, 
                pred_len: int=150,
                padding: int=0,
                factor: int=5, 
                d_model: int=512, 
                n_heads: int=8, 
                e_layers: int=3, 
                d_layers: int=2, 
                d_ff: int=512, 
                dropout: float=0.05, 
                attn: str='prob', 
                activation: str='gelu', 
                output_attention: bool=False, 
                distil: bool=True, 
                mix: bool=True, 
                class_weights: List[float]=[3.,1.],
                max_rmse_err: float=70., 
                learning_rate: float=1e-3, 
                freeze_encoder: bool=False):
        super().__init__()
        self.save_hyperparameters()
        # Scaler
        self.scaler = StandardScaler()
        # Loss function
        self.fi_criterion = nn.CrossEntropyLoss(torch.Tensor(class_weights))
        # Metrics
        self.fi_metric = torchmetrics.AUROC(num_classes=2, average="weighted")
        # Encoding
        self.enc_embedding = GazeEmbedding(enc_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.decoder = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, enc_self_mask=None):
        #x_enc: (bs, sequence_len,2)
        #x_dec: (bs, sequence_len)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(enc_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.hparams.output_attention:
            return dec_out, attns
        else:
            return dec_out

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")
    
    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="val")
    
    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="test")

    def _step(self, batch, batch_idx, step_type):
        try:
            X, fix_y = batch
        except ValueError as e:
            print(f"{batch}")
            raise e

        if self.hparams.output_attention:
            logits, attns = self(X)
        else:
            logits = self(X)
        logits = logits.squeeze().reshape(-1,2) # make long for whole batch
        targets = fix_y.reshape(-1).long() # make 1D vector for whole batch
        #mask = torch.any(X == -180, dim=1)
        loss = self.fi_criterion(logits, targets)
        preds = self._get_preds(logits)
        probs = self._get_probs(logits)
        targets = targets.int()
        auroc = self.fi_metric(probs, targets)
        self.logger.experiment.add_scalars("losses", {f"{step_type}": loss}, self.current_epoch)        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss        

    def configure_optimizers(self):
        params = self.decoder.parameters() if self.hparams.freeze_encoder else list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_probs(self, logits):
        if self.hparams.c_out == 1:
            probs = torch.sigmoid(logits)
        elif self.hparams.c_out > 1:
            probs = torch.softmax(logits, dim=-1)        
        return probs

    def _get_preds(self, logits, threshold=0.5):
        if self.hparams.c_out == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        elif self.hparams.c_out > 1:
            preds = torch.argmax(logits, dim=-1)
        return preds
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InformerEncoderDecoderModel")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--sequence_length', type=int, default=250)
        parser.add_argument('--label_len', type=int, default=100, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=150, help='prediction sequence length')
        parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=2, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        parser.add_argument('--max_rmse_err', type=float, default=70., help='clamps max rmse loss')
        return parser

class InformerMultiTaskEncoderDecoder(LightningModule):
    def __init__(self,
                tasks: List[str]=["fi", "pc", "cl", "rc"],
                enc_in: int=2, 
                dec_in: int=1, 
                c_out: int=2, # output layer size for pretraining fixation classifier 
                seq_len: int=500, 
                label_len: int=100, 
                pred_len: int=150,
                padding: int=0,
                factor: int=5, 
                d_model: int=512, 
                n_heads: int=8, 
                e_layers: int=3, 
                d_layers: int=2, 
                d_ff: int=512, 
                dropout: float=0.05, 
                attn: str='prob', 
                activation: str='gelu', 
                output_attention: bool=False, 
                distil: bool=True, 
                mix: bool=True, 
                class_weights: List[float]=[3.,1.],
                learning_rate: float=1e-3, 
                freeze_encoder: bool=False):
        super().__init__()
        self.save_hyperparameters()
        if len(tasks) == 0:
            raise ValueError("There must be at least one task. Length of tasks is 0")
        # Scaler
        self.scaler = StandardScaler()
        # Encoder
        self.encoder = InformerEncoder(enc_in, factor, d_model, n_heads, e_layers, d_ff, dropout, attn, activation, output_attention, distil)
        # Decoders, metrics, and criterions for each task
        decoders = []
        if "fi" in tasks:
            self.fi_decoder = nn.Linear(d_model, 2, bias=True)
            decoders.append(self.fi_decoder)
            self.fi_criterion = nn.CrossEntropyLoss(torch.Tensor(class_weights))
            #self.fi_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(class_weights))
            self.fi_metric = torchmetrics.AveragePrecision(num_classes=2)
        if "fm" in tasks:
            if "fi" in tasks:
                raise Exception("You can only use one of 'fi' and 'fm' as pretraining tasks, both were provided")
            self.fm_decoder = nn.Linear(d_model, c_out, bias=True)
            decoders.append(self.fm_decoder)
            self.fm_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
            self.fm_metric = torchmetrics.AveragePrecision(num_classes=c_out, average=None, thresholds=20)
        if "pc" in tasks:
            self.pc_decoder = InformerDecoder(dec_in, 2, factor, d_model, n_heads, d_layers, d_ff, dropout, attn, activation, mix)
            decoders.append(self.pc_decoder)
            self.pc_criterion = RMSELoss()
            self.pc_metric = torchmetrics.MeanSquaredError(squared=False)
        if "cl" in tasks:
            #cl_input_dim = hidden_dim * 2
            self.cl_decoder = ae.MLP(input_dim=d_model,
                                layers=[128, 2],
                                activation="relu",
                                batch_norm=True)
            decoders.append(self.cl_decoder)
            self.cl_criterion = nn.CrossEntropyLoss()
            self.cl_metric = torchmetrics.Accuracy(num_classes=2)

            # self.criterions["cl"] = nn.CrossEntropyLoss()
            # self.metrics["cl"] = torchmetrics.Accuracy(num_classes=num_classes)
        if "rc" in tasks:
            #self.decoders["rc"] = create_decoder(hidden_dim,output_seq_length=sequence_length)
            self.rc_decoder = InformerDecoder(dec_in, 2, factor, d_model, n_heads, d_layers, d_ff, dropout, attn, activation, mix)
            decoders.append(self.rc_decoder)
            self.rc_criterion = RMSELoss()
            self.rc_metric = torchmetrics.MeanSquaredError(squared=False)
            # self.criterions["rc"] = RMSELoss()
            # self.metrics["rc"] = torchmetrics.MeanSquaredError()         
        self.decoders = nn.ModuleList(decoders)
        self.c_out=c_out
        
    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #x_enc: (bs, sequence_len,2)
        #x_dec: (bs, sequence_len)
        if self.hparams.output_attention:
            enc_out, attns = self.encoder(x_enc, enc_self_mask)
            dec_out = self.decoder(enc_out, x_dec, dec_self_mask, dec_enc_mask)
            return dec_out, attns
        else:
            enc_out = self.encoder(x_enc, enc_self_mask)
            dec_out = self.decoder(enc_out, x_dec, dec_self_mask, dec_enc_mask)
            return dec_out

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")
    
    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="val")
    
    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, step_type="test")

    def _step(self, batch, batch_idx, step_type):
        try:
            X, fix_y, X2, cl_y = batch
        except ValueError as e:
            try:
                X, fix_y = batch
            except ValueError as e:
                print(f"{batch}")
                raise e
        total_loss = 0
        for task in self.hparams.tasks: # TODO: pass these as a list of models each with own class defined separately
            if task == "cl":
                enc1 = self.encoder(X)
                enc1 = enc1.mean(dim=1)
                enc2 = self.encoder(X2)
                enc2 = enc2.mean(dim=1)
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
                targets_fi = fix_y.reshape(-1).long()
                task_loss = self.fi_criterion(logits, targets_fi)
                probs = self._get_probs(logits)
                task_metric = self.fi_metric(probs, targets_fi.int())
                del enc, probs, targets_fi, fix_y
            elif task == "fm":
                enc = self.encoder(X)
                logits = self.fm_decoder(enc).squeeze().reshape(-1,self.c_out) #TODO: reshape this differently or not at all? 
                # print(f'logits shape {logits.shape}') #logits shape torch.Size([32, 500, 3])
                targets_fm = fix_y.reshape(-1).long() # remove reshape #TODO: reshape to be long, but unreshape the logits
                # print(f'targets shape {targets_fm.shape}')#targets shape torch.Size([32, 500])
                task_loss = self.fm_criterion(logits, targets_fm)
                probs = self._get_probs(logits)
                task_metric = self.fm_metric(probs, targets_fm)
                del enc, probs, targets_fm, fix_y
            elif task == "pc":
                X_pc, y_pc = predictive_coding_batch(X, self.hparams.seq_len, self.hparams.pred_len, self.hparams.label_len)
                enc = self.encoder(X_pc)
                logits = self.pc_decoder(enc, y_pc, pred_len=self.hparams.pred_len).squeeze()[0] \
                     if self.hparams.output_attention else \
                        self.pc_decoder(enc, y_pc, pred_len=self.hparams.pred_len).squeeze()
                # print(f'logits_pc: {logits.shape}')
                # print(f'y_pc: {y_pc.shape}')
                assert(logits.shape == y_pc.shape)
                mask = y_pc > -180
                task_loss = self.pc_criterion(logits[mask], y_pc[mask])
                logits = self.scaler.inverse_transform(logits)
                y_pc = self.scaler.inverse_transform(y_pc)
                task_metric = self.pc_metric(logits[mask], y_pc[mask])
                #task_loss = torch.clamp(self.pc_criterion(logits, y_pc), max=self.hparams.max_rmse_err)
                #task_metric = self.pc_metric(logits, y_pc)
                del X_pc, y_pc
            elif task == "rc":
                enc = self.encoder(X)
                logits = self.rc_decoder(enc, X).squeeze()
                mask = X > -180
                task_loss = self.pc_criterion(logits[mask], X[mask])
                logits = self.scaler.inverse_transform(logits)
                y_rc = self.scaler.inverse_transform(X)
                task_metric = self.pc_metric(logits[mask], y_rc[mask])
                #task_loss = torch.clamp(self.rc_criterion(logits, X), max=self.hparams.max_rmse_err)
                #task_metric = self.rc_metric(logits, X)
                del y_rc               
            else:
                raise ValueError("Task not recognized.")
            self.log(f"{step_type}_{task}_loss", task_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{step_type}_{task}_metric", task_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            total_loss += task_loss
        self.log(f"{step_type}_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):
        params = [list(m.parameters()) for m in self.decoders]
        params.append(list(self.encoder.parameters()))
        params = reduce(lambda x,y: x + y, params)
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_probs(self, logits):
        if self.hparams.c_out == 1:
            probs = torch.sigmoid(logits)
        elif self.hparams.c_out > 1:
            probs = torch.softmax(logits, dim=1)        
        return probs

    def _get_preds(self, logits, threshold=0.5):
        if self.hparams.c_out == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        elif self.hparams.c_out > 1:
            preds = torch.argmax(logits, dim=1)
        return preds
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InformerMultiTaskModel")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--sequence_length', type=int, default=250)
        parser.add_argument('--label_len', type=int, default=100, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=150, help='prediction sequence length')
        parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=2, help='output size/nclass for fixations')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--tasks', type=str, nargs='*', default=["fi", "cl", "rc", "pc"])
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        return parser
    
class InformerClassifierModel(LightningModule):
    """Informer encoder-classifier stack for binary classification of entire sequence

    Args:
        LightningModule (_type_): _description_
    """    
    def __init__(self, 
                enc_in: int=2, 
                factor: int=5, 
                d_model: int=512, 
                n_heads: int=8, 
                e_layers: int=3, 
                d_ff: int=512, 
                dropout: float=0.05, 
                attn: str='prob', 
                activation: str='gelu', 
                output_attention: bool=False, 
                distil: bool=True, 
                learning_rate: float=1e-3, 
                encoder_ckpt: str="",
                freeze_encoder: bool=False):
        super().__init__()
        self.save_hyperparameters()
        # Scaler
        self.scaler = StandardScaler()
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        # Metrics
        self.auroc_metric = torchmetrics.AUROC()
        self.accuracy_metric = torchmetrics.Accuracy()
        # Encoding
        if encoder_ckpt:
            #self.enc_embedding, self.encoder = get_encoder_from_checkpoint(InformerMultiTaskEncoderDecoder, encoder_ckpt)
            self.encoder = get_encoder_from_checkpoint(InformerMultiTaskEncoderDecoder, 
                                                       encoder_ckpt)
        else:
            self.enc_embedding = GazeEmbedding(enc_in, d_model, dropout)
            #self.dec_embedding = GazeEmbedding(dec_in, d_model, dropout)
            # Attention
            Attn = ProbAttention if attn=='prob' else FullAttention
            # Encoder

            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, 
                                            factor, 
                                            attention_dropout=dropout, 
                                            output_attention=output_attention), 
                                            d_model, 
                                            n_heads, 
                                            mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(e_layers-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        self.classifier_head = ae.MLP(input_dim=d_model, 
                                      layers=[64,1], 
                                      activation="relu")

        if freeze_encoder:
            #self.enc_embedding.requires_grad_(False)
            self.encoder.requires_grad_(False)

    def forward(self, x_enc, enc_self_mask=None):
        #enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(x_enc, enc_self_mask)
        dec_in = torch.mean(enc_out, 1)
        dec_out = self.classifier_head(dec_in)
        if self.hparams.output_attention:
            #return dec_out, attns
            return dec_out
        else:
            return dec_out
        
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
        logits = self(X).squeeze()
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
        #params = self.classifier_head.parameters() if self.hparams.freeze_encoder else list(self.enc_embedding.parameters()) + list(self.encoder.parameters()) + list(self.classifier_head.parameters())
        params = self.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_preds(self, logits, threshold=0.5):
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        return preds

    def _get_probs(self, logits):
        probs = torch.sigmoid(logits)
        return probs

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InformerEncoderDecoderModel")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--encoder_ckpt', type=str, default="")
        parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
        parser.add_argument('--c_out', type=int, default=1, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--freeze_encoder', action='store_false')
        return parser



class InformerEncoderMulticlassModel(InformerEncoderFixationModel):
#     """Informer encoder-decoder stack for multiclass classification of each sample in the sequence
#     Generalisation of InformerEncoderFixationModel for >2 classes
#     """    
    def __init__(self, n_classes):
        super.__init__(c_out=n_classes, class_weights=[1.]*n_classes)

        # Loss function
        self.fm_criterion = nn.CrossEntropyLoss(torch.Tensor(class_weights))
        # Metrics
        self.fm_metric = torchmetrics.AveragePrecision(num_classes=c_out, average="weighted")


    def _step(self, batch, batch_idx, step_type):
            try:
                X, fix_y = batch
            except ValueError as e:
                print(f"{batch}")
                raise e

            if self.hparams.output_attention:
                logits, attns = self(X)
            else:
                logits = self(X)
            logits = logits.squeeze().reshape(-1,2) # make long for whole batch
            targets = fix_y.reshape(-1).long() # make 1D vector for whole batch
            #mask = torch.any(X == -180, dim=1)
            loss = self.fm_criterion(logits, targets)
            preds = self._get_preds(logits)
            probs = self._get_probs(logits)
            targets = targets.int()
            metric = self.fm_metric(probs, targets)
            self.logger.experiment.add_scalars("losses", {f"{step_type}": loss}, self.current_epoch)        
            self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{step_type}_metric", metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return loss   
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InformerEncoderDecoderModel")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--sequence_length', type=int, default=250)
        parser.add_argument('--label_len', type=int, default=100, help='start token length of Informer decoder')
        parser.add_argument('--pred_len', type=int, default=150, help='prediction sequence length')
        parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=3, help='output size, n classes')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--class_weights', type=float, nargs='*', default=[1., 1., 1.], help = 'weights per class to use for loss function, list of length c_out')
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        parser.add_argument('--max_rmse_err', type=float, default=70., help='clamps max rmse loss')
        return parser





class InformerEncoderScalarRegModel(LightningModule):
    """Informer encoder-regression head stack for regression of entire sequence to scalar label;

    Args:
        LightningModule (_type_): _description_
    """    
    def __init__(self, 
                enc_in: int=2, 
                factor: int=5, 
                d_model: int=512, 
                n_heads: int=8, 
                e_layers: int=3, 
                d_ff: int=512, 
                dropout: float=0.05, 
                attn: str='prob', 
                activation: str='gelu', 
                output_attention: bool=False, 
                distil: bool=True, 
                learning_rate: float=1e-3, 
                encoder_ckpt: str="",
                freeze_encoder: bool=False):
        super().__init__()
        self.save_hyperparameters()
        # Scaler
        self.scaler = StandardScaler()
        # Loss function
        self.criterion = RMSELoss() 
        # Metrics
        self.accuracy_metric = torchmetrics.MeanAbsoluteError()
        # Encoding
        if encoder_ckpt:
            #self.enc_embedding, self.encoder = get_encoder_from_checkpoint(InformerMultiTaskEncoderDecoder, encoder_ckpt)
            self.encoder = get_encoder_from_checkpoint(InformerMultiTaskEncoderDecoder, 
                                                       encoder_ckpt)
        else:
            self.enc_embedding = GazeEmbedding(enc_in, d_model, dropout)
            #self.dec_embedding = GazeEmbedding(dec_in, d_model, dropout)
            # Attention
            Attn = ProbAttention if attn=='prob' else FullAttention
            # Encoder

            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, 
                                            factor, 
                                            attention_dropout=dropout, 
                                            output_attention=output_attention), 
                                            d_model, 
                                            n_heads, 
                                            mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(e_layers-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        self.classifier_head = ae.MLP(input_dim=d_model, 
                                      layers=[64,1], 
                                      activation="relu")

        if freeze_encoder:
            #self.enc_embedding.requires_grad_(False)
            self.encoder.requires_grad_(False)

    def forward(self, x_enc, enc_self_mask=None):
        #enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(x_enc, enc_self_mask)
        dec_in = torch.mean(enc_out, 1)
        dec_out = self.classifier_head(dec_in)
        if self.hparams.output_attention:
            #return dec_out, attns
            return dec_out
        else:
            return dec_out
        
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
        logits = self(X).squeeze()
        loss = self.criterion(logits, y)
        probs = self._get_probs(logits)
        accuracy = self.accuracy_metric(probs, y)
        auroc = self.auroc_metric(probs, y)
        self.logger.experiment.add_scalars("losses", {f"{step_type}_loss": loss}, self.current_epoch)        
        self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        #params = self.classifier_head.parameters() if self.hparams.freeze_encoder else list(self.enc_embedding.parameters()) + list(self.encoder.parameters()) + list(self.classifier_head.parameters())
        params = self.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)
        res = {"optimizer": optimizer}
        res['lr_scheduler'] = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(self.trainer.max_epochs / 5)), gamma=0.5)}
        return res

    def _get_preds(self, logits, threshold=0.5):
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        return preds

    def _get_probs(self, logits):
        probs = torch.sigmoid(logits)
        return probs

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InformerEncoderDecoderModel")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--encoder_ckpt', type=str, default="")
        parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
        parser.add_argument('--c_out', type=int, default=1, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
        parser.add_argument('--padding', type=int, default=0, help='padding type')
        parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
        parser.add_argument('--activation', type=str, default='gelu',help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--class_weights', type=float, nargs='*', default=[3., 1.])
        parser.add_argument('--freeze_encoder', action='store_false')
        return parser