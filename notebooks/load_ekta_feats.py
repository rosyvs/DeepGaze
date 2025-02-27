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

# %%
emb_file = os.path.normpath(os.path.join(repodir,"./data/embeddings_for_rosy_DST/EML_PTEZ_FTEML_Embeddings_Script_from_long_ago/EML_split4_embeddings.npy"))
label_file =  os.path.normpath(os.path.join(repodir,"./data/EML/EML1_pageLevel_500+_matchEDMinstances.csv"))
label_df = pd.read_csv(label_file, keep_default_na=False)


# read the file
embeddings = np.load(emb_file)
embeddings['name']
embeddings['task']
embeddings['embedding'].shape # n, len, feats
embeddings.dtype.names
embeddings[0]['original_data'].shape # len, 3

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
