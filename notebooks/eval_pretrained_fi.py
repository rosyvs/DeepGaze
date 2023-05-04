#%% 
import sys
from pathlib import Path
OBF_RELPATH = "../../OBF"
sys.path.append(str(Path(OBF_RELPATH).resolve()))
from eyemind.obf.model import ae
from eyemind.obf.model import creator
from eyemind import obf
from eyemind.dataloading.load_dataset import limit_sequence_len, get_label_mapper, get_filenames_for_dataset, create_filename_col, get_stratified_group_splits
from eyemind.dataloading.gaze_data import GazeDataModule
from eyemind.models.classifier import EncoderClassifierModel
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler, DataLoader
from pytorch_lightning import Trainer

#%% [markdown]
### load pretrained OBF RNNgru encoder and FI decoder and run it on our data

#%% paths
data_folder = Path("../data/processed/fixation")
label_filepath = Path("../data/processed/EML1_pageLevel_500+_matchEDMinstances.csv") 
pre_trained_weights_dir = Path(OBF_RELPATH + "/pre_weights/sample_weights/")
folds = Path("data_splits/4fold_participant/seed21.yml")

# %% load pretrained encoder
encoder = creator.load_encoder(str(pre_trained_weights_dir.resolve()))

#%% load pretrained decoder
fi_decoder = torch.load(str(Path(pre_trained_weights_dir, "fi_1633040995_gru.pt").resolve()),map_location=torch.device('cpu'))
class FixationIdentifier(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        embeddings = self.encoder(x)
        output = self.decoder(embeddings)
        return output
fid_model = FixationIdentifier(encoder, fi_decoder)


#%% set up data 
# Read the labels and create id
label_cols = ["Rote_X", "Inference_X", "Deep_X", "Rote_Y", "Inference_Y", "Rote_Z", "Inference_Z", "Deep_Z", "Rote_D", "Inference_D","MW"]
label_df = pd.read_csv(label_filepath)
l_ds = get_datasets(label_cols, label_df, data_folder, x_transforms=[limit_sequence_len,lambda data: torch.tensor(data).float()], y_transforms=[lambda data: torch.tensor(data).float()])

#%% dataloaders



#%% evaluate OBF pretrained on FID


#%% train classifier head on pretrained OBF

#%% Train OBF GRU for FI
# useful code potentially:
# experiments/cross_val_fixation.py
# experimenter.py get_encoder_classifier_model 
# fixatoin_experiment.py seems to load the obf ae models
trainer = Trainer(max_epochs=10, logger=logger)
trainer.fit(model, train_dl, val_dl)
trainer.test(model,
             dataloader=test_dl,
            ckpt_path=)
#%% Run comprehension OBF GRU for FI