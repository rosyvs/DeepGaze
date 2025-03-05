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
from eyemind.dataloading.limu_bert_loader import GazeformerEmbeddingDataset, gazeformer_embedding_collate_fn, EmbeddingDataModule
from eyemind.models.classifier import ClassifierHead
from eyemind.analysis.predictions import get_encoder_from_checkpoint
import matplotlib.pyplot as plt
from eyemind.analysis.visualize import plot_scanpath_labels, viz_coding, fixation_image, plot_scanpath_pc
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))
import numpy as np
from functools import partial
from pytorch_lightning.cli import LightningCLI
import numpy as np

# # %%
# ScanEZ embeddings (I extracted train and val) 
#     Train: new_split{1,2,3,4}_embeddings.npy
#      Val:  new_split{1,2,3,4}_embeddings_val.npy

# W/o Pretraining (e.g., only trained on human data)  
#      Train: new_Only_split{12,3,4}_embeddings.npy
#      Val: new_Only_split{1,2,3,4}_embeddings_val.npy

# W/o Finetuning (e.g., only trained on synthetic data) 
#      Train: newEZ_Reader_Only_split{1,2,3,4}_embeddings.npy 
#      Val: newEZ_Reader_Only_split{12,3,4}_embeddings_val.npy

# The data inside eml_structured_21 and eml_unstructured_21 are the npy files used to train/finetune the LIMU model 

# %%
emb_file = os.path.normpath(os.path.join(repodir,'./data/ekta_embeddings/new_split2_embeddings.npy/rosie_train_2_with_embedding_usingnew_e_pretrain_200_FT_EML_sentid_split_fold2.npy'))
label_file =  os.path.normpath(os.path.join(repodir,"./data/EML/EML1_pageLevel_500+_matchEDMinstances.csv"))
label_df = pd.read_csv(label_file, keep_default_na=False)


# read the file
embeddings = np.load(emb_file)
print(f'embeddings fields: {embeddings.dtype.names}')
print(embeddings['embedding'].shape) # n, len, feats
embeddings[0]['original_data'].shape # len, 3
print(f'len embeddings: {len(embeddings)}')

# read label file
label_df = pd.read_csv(label_file, keep_default_na=False)
print(f'len label_df: {len(label_df)}')
#%%
def get_len(x, pad_val=-1.0):
    # len, 3
    xi = x['original_data']
    # length of sequence is up to where no value is equal to pad val in the 2nd dim
    true_ix=np.where((xi!=pad_val).sum(axis=1)==xi.shape[1])
    true_length = true_ix[0][-1]+1
    return true_length
get_len(embeddings[np.random.randint(0, embeddings.shape[0])])
# indices are events

#%%
ds = GazeformerEmbeddingDataset(emb_file, label_file, max_sequence_length=125, label_col='Rote_X')
collate_fn = partial(gazeformer_embedding_collate_fn)
dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)

# dl_default = torch.utils.data.DataLoader(ds, batch_size=1)

#%% check for short sequences in dataloader
max_len = 0
min_len = 99999
for i, emb in enumerate(embeddings):
    # check emv has 2 dims
    if len(emb['embedding'].shape) != 2:
        print(f'found embedding with shape {emb["embedding"].shape} at index {i}')
    if get_len(emb) > max_len:
        max_len = get_len(emb)
    if get_len(emb) < min_len:
        min_len = get_len(emb)
    if get_len(emb) < 10:
        print(f'found short sequence at index {i} ({100*i/len(embeddings):.2f}%) with length {get_len(emb)}')
        print(emb['embedding'].shape)
        print(emb['original_data'].shape)
print(f'max len: {max_len}, min len: {min_len}')

#%% iterate dataset
# for i, batch in enumerate(ds):
#     if i>10:
#         break
#     print(batch)
# %%
# for i, batch in enumerate(dl):
#     if i>10:
#         break
#     print(batch)
# # %%
# for i, batch in enumerate(dl_default):
#     if i>10:
#         break
#     print(batch)

# %% train classifier head on embeddings
# %%


cli = LightningCLI(ClassifierHead, 
                        EmbeddingDataModule, 
                        run=False, 
                        save_config_overwrite=True,
                        )
cli.datamodule.setup()
cli.trainer.fit(cli.model, datamodule=cli.datamodule)
# %%
