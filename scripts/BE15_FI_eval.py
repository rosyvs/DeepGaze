#%%
import pandas as pd
import yaml
import torch
import os
from pytorch_lightning import Trainer, seed_everything
from eyemind.trainer.loops import KFoldLoop
import eyemind
from eyemind.models.transformers import InformerEncoderDecoderModel, InformerEncoderFixationModel, InformerMultiTaskEncoderDecoder
from eyemind.dataloading.informer_data import InformerDataModule
from eyemind.analysis.visualize import fixation_image
import matplotlib.pyplot as plt
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))

#%% Dataloader for BE15
datamodule = InformerDataModule(
    data_dir=os.path.join(repodir,'data/BE15/gaze+fix'), 
    label_filepath=os.path.join(repodir,'data/BE15/BE15_instances.csv'), # this determines wihch instanecs are used 
    load_setup_path=None,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    train_fold=None,
    val_fold=None,
    batch_size=1,
    contrastive=False,
    sequence_length=500,
    min_scanpath_length=500,
    label_length=0, # ?? default is 48 why?
    pin_memory=False,
    num_workers=1
    )

datamodule.setup()
test_dl = datamodule.get_dataloader(datamodule.test_dataset) # this is the held out fold's dataloader
DEBUG=True
print(f'test data: {len(test_dl.dataset.dataset.files) } files')
print(f'test data: {len(test_dl)} instances loaded')


#%% Load model trained on EML
auprcs=[]
for fold in [0, 1, 2, 3]:
# for fold in [0]:
    fi_targets_all=[]
    fi_probs_all=[]
    save_dir = f"{repodir}/lightning_logs/informer_pretraining_seed21/fold{fold}/"
    config_path=os.path.join(save_dir,"config.yaml")
    ckpt_path = os.path.join(save_dir,"checkpoints","last.ckpt")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    seed_everything(config["seed_everything"], workers=True) # not sure if this is needed

    model = InformerMultiTaskEncoderDecoder.load_from_checkpoint(ckpt_path,
                                                    # encoder_weights_path=None
                                                    )
    encoder=model.encoder
    decoder=model.fi_decoder
    model.eval()
    encoder.eval()
    decoder.eval()
    print(f'test data: {len(test_dl)} instances')

    for i,batch in enumerate(test_dl):
        if DEBUG:
            print(f"batch: {i}")
            print(f"batch size: {len(batch[0])}")
        with torch.no_grad():
            # preds = model(batch) # this step fails
            logits=encoder(batch[0], None)
            fi_logits = model.fi_decoder.forward(logits)
            fi_preds=fi_logits.max(2).indices
            fi_targets = batch[1]
            fi_probs = model._get_probs(fi_logits)
            fi_logits_long = fi_logits.squeeze().reshape(-1,2)
            fi_targets_long = fi_targets.reshape(-1).long()
            fi_probs_long = model._get_probs(fi_logits_long)
        if DEBUG:
            # or reshape batch into one long vector in Ricks's code, to get batch-wisemetric:
            # pick one from batch to plot
            one_pred = fi_preds[0,:]
            one_target = fi_targets[0,:]
            print(one_pred.shape)
            print(one_target.shape)

            fixation_image(one_pred, one_target)
        #mask = torch.any(X == -180, dim=1)
        loss = model.fi_criterion(fi_logits_long, fi_targets_long)
        auprc = model.fi_metric(fi_probs_long, fi_targets_long.int())

        if DEBUG:
            print(f'batch AUPRC: {auprc:.3f}')
        if ((i==4) and DEBUG) : # just run a couple to check
            break
        fi_targets_all.append(fi_targets_long.int())
        fi_probs_all.append(fi_probs_long)

    fi_targets_all2 = torch.cat(fi_targets_all)
    fi_probs_all = torch.vstack(fi_probs_all)

    FOLDauprc = model.fi_metric(fi_probs_all, fi_targets_all2)
    print(f'fold {fold}: FI AUPRC {FOLDauprc}')
    auprcs.append(float(FOLDauprc))

print(f'mean AUPRC: {sum(auprcs)/len(auprcs):.3f}')# %% 
