from typing import List
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from eyemind.dataloading.batch_loading import fixation_batch, predictive_coding_batch
from eyemind.models.informer.models.model import InformerStack

from eyemind.models.informer.utils.masking import TriangularCausalMask, ProbMask
from eyemind.models.informer.models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from eyemind.models.informer.models.decoder import Decoder, DecoderLayer
from eyemind.models.informer.models.attn import FullAttention, ProbAttention, AttentionLayer
from eyemind.models.informer.models.embed import GazeEmbedding
from eyemind.models.loss import RMSELoss

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
        #x_enc: (bs, sequence_len,2)
        #x_dec: (bs, sequence_len)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #print(x_dec[0])
        dec_out = self.dec_embedding(x_dec)
        #print(dec_out[0])
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
        task_metric = self.pc_metric(logits[mask], Y_pc[mask])
        self.log(f"{step_type}_loss", task_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_type}_pc_metric", task_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return task_loss
        # Task 1. Fixation ID
        # fix_decoder_inp, targets = fixation_batch(self.hparams.seq_len, self.hparams.label_len, self.hparams.pred_len, X, fix_y, padding=self.hparams.padding)
        # if self.hparams.output_attention:
        #     logits = self(X, fix_decoder_inp)[0]
        # else:
        #     logits = self(X, fix_decoder_inp)
        # logits = logits.squeeze().reshape(-1,2)
        # targets = targets.reshape(-1).long()
        # loss = self.criterion(logits, targets)
        # preds = self._get_preds(logits)
        # probs = self._get_probs(logits)
        # targets = targets.int()
        # accuracy = self.accuracy_metric(probs, targets)
        # auroc = self.auroc_metric(probs, targets)
        # self.logger.experiment.add_scalars("losses", {f"{step_type}": loss}, self.current_epoch)        
        # self.log(f"{step_type}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"{step_type}_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"{step_type}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # return loss

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
