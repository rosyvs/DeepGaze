import yaml
from pathlib import Path
from eyemind.dataloading.gaze_data import BaseSequenceToSequenceDataModule
import torch

def load_model_from_checkpoint(model_cls, checkpoint_path):
    return model_cls.load_from_checkpoint(checkpoint_path)

# def get_encoder_from_checkpoint(model_cls, checkpoint_path):
#     full_model = model_cls.load_from_checkpoint(checkpoint_path)
#     return full_model.enc_embedding, full_model.encoder

def get_encoder_from_checkpoint(model_cls, checkpoint_path):
    full_model = model_cls.load_from_checkpoint(checkpoint_path)
    return full_model.encoder

def get_dataloader(config_path, dm_cls=BaseSequenceToSequenceDataModule, data_base_dir="/Users/rickgentry/emotive_lab/eyemind/data", label_filepath="processed/EML1_pageLevel_with_filename_seq.csv", data_dir="processed/fixation"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["data_dir"] = str(Path(data_base_dir, data_dir).resolve())
    config["data"]["label_filepath"] = str(Path(data_base_dir, label_filepath).resolve())
    dm = dm_cls(**config['data'])
    dm.setup()
    return dm.predict_dataloader()

def fixation_preds_targets(model, dl, samples=1):
    model.eval()
    batch = next(iter(dl))
    logits = model.forward(batch[0])
    fixation_preds = model._get_preds(logits)
    fixation_targets = batch[1]
    rows = torch.randint(high=len(fixation_preds),size=(samples,))
    return fixation_preds[rows].squeeze(), fixation_targets[rows].squeeze()

def tasks_preds_targets(model, dl, samples=1):
    model.eval()
    batch = next(iter(dl))
    model.predict_step(batch, 0)
    