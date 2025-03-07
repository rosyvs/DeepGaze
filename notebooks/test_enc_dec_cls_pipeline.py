#%%
import pandas as pd
import yaml
import torch
import os
from pytorch_lightning import Trainer, seed_everything
from eyemind.trainer.loops import KFoldLoop
import eyemind
from eyemind.models.transformers import InformerEncoder, InformerEncoderDecoderModel, InformerEncoderFixationModel, InformerMultiTaskEncoderDecoder, InformerClassifierModel
from eyemind.dataloading.informer_data import InformerDataModule, InformerMultiLabelDatamodule,  InformerVariableLengthDataModule
from eyemind.dataloading.gaze_data import VariableLengthSequenceToLabelDataModule
from eyemind.models.classifier import ClassifierHead
from eyemind.analysis.predictions import get_encoder_from_checkpoint
import matplotlib.pyplot as plt
from eyemind.analysis.visualize import plot_scanpath_labels, viz_coding, fixation_image, plot_scanpath_pc

#%% LOAD INFORMER ENCODER CHECKPOINT
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))
# save_dir_base = f"{repodir}/lightning_logs/2024/cluster/new_multitask_informer_pretraining"
save_dir_base = f"{repodir}/lightning_logs/2023/informer_pretraining_seed21"
is_old_version=True # i.e. before I made changes to model and datamodule to support multiple labels etc
fold=0
save_dir = os.path.join(save_dir_base, f'fold{fold}/')
config_path=os.path.join(save_dir,"config.yaml")
ckpt_path = os.path.join(save_dir,"checkpoints","last.ckpt")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# Load just encoder
encoder=get_encoder_from_checkpoint(InformerMultiTaskEncoderDecoder, ckpt_path)
encoder.eval()

#%% set up a multitask InformerDataModule to load the same data as used in training
test_data_dir = os.path.join(repodir,"data/processed/fixation")
test_label_file =  os.path.join(repodir,"./data/EML/EML1_pageLevel_500+_matchEDMinstances.csv")
multitask_data_cfg= config["data"].copy()
multitask_data_cfg["data_dir"]=test_data_dir
multitask_data_cfg["label_filepath"]=test_label_file

# edit config for consistency woth the new version of the datamodule
if is_old_version:
    multitask_data_cfg["min_sequence_length"] = multitask_data_cfg.pop("min_scanpath_length", 500)
    multitask_data_cfg["sample_label_col"]="fixation_label"
    multitask_data_cfg["file_label_col"]=None
    multitask_data_cfg["usecols"] = [2,3]
datamodule = InformerMultiLabelDatamodule(**multitask_data_cfg )
datamodule.setup()
mlt_test_dl = datamodule.get_dataloader(datamodule.test_dataset) # this is the held out fold's dataloader
for i,batch in enumerate(mlt_test_dl):
    n_items = len(batch) # this is not the batch size, but the number of items (data and labels) to unpack
    if n_items==2: # just gaze sequence and fixation (sample) labels
        X, yi = batch
        X, X_mask = X 
        yi, yi_mask = yi
    elif n_items==4: # contrastive so X and X2 are present
        X, yi, X2, cl_y = batch
        X, X_mask = X 
        yi, yi_mask = yi
        X2, X2_mask = X2
    elif n_items==5: # sequence and fixation (sample) labels
        X, yi, seq_y, X2, cl_y = batch
        X, X_mask = X
        yi, yi_mask = yi
        seq_y, seq_y_mask = seq_y
        X2, X2_mask = X2
    with torch.no_grad():
        X_enc=encoder(X, None)
    if i==0: # just run a couple to check
        break
perc_nan = torch.isnan(X_enc).sum() / torch.numel(X_enc)
print(f'percentage of nans in  encoder X_enc: {100*perc_nan}')
print(f'X_enc shape: {X_enc.shape}')



#%% TEST USING VARIABLE LENGTH MULTITASK DATALOADER #TODO: NOT WRITTERN DATALOADER CLASS YET...


#%% TEST USING VARIALBE LENGTH SEQUENCE TO LABEL DATALOADER
from eyemind.dataloading.informer_data import VariableLengthSequenceToLabelDataModule
var_data_config = multitask_data_cfg.copy()
file_label_col='Rote_X'
var_data_config.update({"min_sequence_length":10, 'max_sequence_length':2000})
# remove invalid args
del var_data_config["label_length"]
del var_data_config["pred_length"]
del var_data_config["contrastive"]
del var_data_config["sample_label_col"]
del var_data_config["sequence_length"]
var_data_config["label_col"] = file_label_col
del var_data_config["file_label_col"]
datamodule = VariableLengthSequenceToLabelDataModule(**var_data_config)
datamodule.setup()
test_dl = datamodule.get_dataloader(datamodule.test_dataset) # this is the held out fold's dataloader
var_data_config

#%% iterate dataloader
for i,batch in enumerate(test_dl):
    print(f"batch: {i}")
    print(f"length of batch: {len(batch)}")
    n_items = len(batch) # this is not the batch size, but the number of items (data and labels) to unpack
    if n_items==2: # just gaze sequence and fixation (sample) labels
        X, y = batch
        X, X_mask = X 
    print(X.shape)
    with torch.no_grad():
        # preds = model(batch) # this step fails
        logits_from_masked=encoder(X, X_mask)
        logits=encoder(X)
        #compare
        print(torch.equal(logits, logits_from_masked))
        perc_nan = torch.isnan(logits).sum() / torch.numel(logits)
        print(f'percentage of nans in  encoder logits: {100*perc_nan:.2f}')
        perc_nan = torch.isnan(logits_from_masked).sum() / torch.numel(logits_from_masked)
        print(f'percentage of nans in  encoder logits from masked: {100*perc_nan:.2f}')
    if i==0: # just run a couple to check
        break


#%% test using variable length sequences as input to encoder-classifier
from eyemind.models.transformers import InformerClassifierModel
save_dir = os.path.join(repodir, f'lightning_logs/2023/informer_{file_label_col}/fold0')
ckpt_path = os.path.join( '..',f'lightning_logs/2023/informer_{file_label_col}/fold0',"checkpoints","last.ckpt")
config_path=os.path.join(save_dir,"config.yaml")

with open(config_path, "r") as f:
    ic_config = yaml.safe_load(f)['model']
ic_config['encoder_ckpt'] = os.path.join(repodir, ic_config['encoder_ckpt'].replace('lightning_logs','lightning_logs/2023'))
# initialise the model
ic_model = InformerClassifierModel.load_from_checkpoint(ckpt_path,**ic_config)
with torch.no_grad():
    logits = ic_model(X, X_mask)
    ic_pred = ic_model._get_probs(logits).squeeze()
print(f'ic_pred: {ic_pred}')
print(f'classifier accuracy: {ic_model.accuracy_metric(ic_pred, y.int())}')

#%% pooling
from eyemind.dataloading.transforms import Pooler
mean_pooler = Pooler('mean')
masked_mean_pooler = Pooler('masked_mean')
final_pos_pooler = Pooler('final_pos')

#%% test using InformerEncoder and ClassifierHead separately
ch_config = ic_config.copy()
unu = ['encoder_ckpt','n_heads','e_layers','attn','distil','factor','enc_in','d_ff','output_attention','freeze_encoder']
for k in unu:
    ch_config.pop(k)
classifier = ClassifierHead(**ch_config)
with torch.no_grad():
    logits=encoder(X, X_mask)
    # pooling
    
    mean_pool_logits = mean_pooler(logits)
    masked_mean_pool_logits = masked_mean_pooler(logits, X_mask)
    final_pos_logits = final_pos_pooler(logits, X_mask)

    masked_mean_pred = classifier(masked_mean_pool_logits).squeeze()
    final_pos_pred = classifier(final_pos_logits).squeeze()
    mean_pool_pred = classifier(mean_pool_logits).squeeze()
# print predictions and accuracy
print(f'mean_pool_pred: {mean_pool_pred}')
print(f'classifier accuracy: {classifier.accuracy_metric(mean_pool_pred, y.int())}')
print(f'final_pos_pred: {final_pos_pred}')
print(f'classifier accuracy: {classifier.accuracy_metric(final_pos_pred, y.int())}')
print(f'masked_mean_pred: {masked_mean_pred}')
print(f'classifier accuracy: {classifier.accuracy_metric(masked_mean_pred, y.int())}')

#%% test loading feats from elsewhere and running classifier head only
# data needs to be encoder feature vector + discrete label of length 1
