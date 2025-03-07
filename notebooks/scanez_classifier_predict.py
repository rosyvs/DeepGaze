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
from eyemind.dataloading.transforms import Pooler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import ttest_rel
import numpy as np

from eyemind.dataloading.limu_bert_loader import GazeformerEmbeddingDataset, gazeformer_embedding_collate_fn, EmbeddingDataModule
from functools import partial
import matplotlib.pyplot as plt
from eyemind.analysis.visualize import plot_scanpath_labels, viz_coding, fixation_image, plot_scanpath_pc
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))

# %% setup
pooling_methods = ["mean", "final_pos", "masked_mean"]
versions = ["ptEZftEML","ptEZ", "ptEML"]
label_cols = ['SVT','Rote_X', 'Rote_Y', "Rote_Z",
          'Inference_X', "Inference_Y","Inference_Z",
          "Deep_X", "Deep_Z",
          "Rote_D","Inference_D",
          "MW"]
folds = [0,1,2,3]

#%% TEMP 
label_col = label_cols[0]
version = versions[0]
pool_method = pooling_methods[0]
classifier_names = [f"{version}_{pool_method}" for version in versions for pool_method in pooling_methods]
classifier_name = classifier_names[0]
fold = folds[0]

res_all = []
for version in versions:
    for pool_method in pooling_methods:
        for label_col in label_cols:
            save_dir_base = f"{repodir}/lightning_logs/2025/classifiers/scanez/{label_col}_scanez_{classifier_name}/"
            for fold in folds:
                print(f"Loading model for {version} {pool_method} {label_col} fold {fold}")
                save_dir = os.path.join(save_dir_base, f'fold{fold}/')
                config_path=os.path.join(save_dir,"config.yaml")
                ckpt_path = os.path.join(save_dir,"checkpoints","last.ckpt")
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                # load the classifier head
                classifier = ClassifierHead.load_from_checkpoint(ckpt_path, **config["model"])
                classifier.eval()

                test_data_cfg = config["data"].copy()
                test_data_cfg["data_path"]=config["data"]["val_data_path"]
                DS = GazeformerEmbeddingDataset(data_path=os.path.join(repodir,test_data_cfg["data_path"]), 
                                                label_filepath=os.path.join(repodir,test_data_cfg["label_filepath"]), 
                                                min_sequence_length=test_data_cfg["min_sequence_length"],
                                                max_sequence_length = test_data_cfg["max_sequence_length"],
                                                label_col=label_col)
                pool_fn = Pooler(pool_method)
                collate_fn = partial(gazeformer_embedding_collate_fn, pool_fn = pool_fn)
                DL = torch.utils.data.DataLoader(DS, batch_size=test_data_cfg["batch_size"], 
                                                num_workers=test_data_cfg["num_workers"],
                                                pin_memory=test_data_cfg["pin_memory"],
                                                collate_fn=collate_fn,
                                                shuffle=False)

                preds = []
                probs = []
                labels = []
                # get ids from datset
                ids = DS.ids
                for i, batch in enumerate(DL):
                    X, y = batch # X is (embedding, mask), y is the label
                    feat = pool_fn(X[0], X[1])
                    with torch.no_grad():
                        y_logits = classifier(X[0])
                        y_prob = classifier._get_probs(y_logits).squeeze()
                        y_pred = classifier._get_preds(y_logits).squeeze()
                    preds.extend(y_pred.tolist())
                    probs.extend(y_prob.tolist())
                    labels.extend(y.tolist())
                diff = np.abs(np.array(probs) - np.array(labels))
                correct = np.array(preds) == np.array(labels)
                res = pd.DataFrame({"version":version, "pool_method":pool_method, "label_col":label_col, 
                                    "id":ids,"fold":fold,"prob":probs,"pred":preds, "label":labels, 
                                    "diff":diff, "correct":correct})
                res_all.append(res)            
res_all = pd.concat(res_all)
res_all['diff'] = np.abs(res_all['label'] - res_all['prob'])
res_all['correct'] = res_all['label'] == res_all['pred']
res_all.to_csv(f"{repodir}/results/scanez_EML_predictions.csv", index=False)



# %% F1 score per model
modelwise_res = []
for version in versions:
    for pool_method in pooling_methods:
        for label_col in label_cols:
            this_res = res_all[(res_all["version"]==version) & (res_all["pool_method"]==pool_method) & (res_all["label_col"]==label_col)]
            f1 = f1_score(this_res["label"], this_res["pred"])
            auroc = roc_auc_score(this_res["label"], this_res["prob"])
            acc = accuracy_score(this_res["label"], this_res["pred"])
            modelwise_res.append({"version":version, "pool_method":pool_method, "label_col":label_col, "F1":f1, "AUROC":auroc, "Accuracy":acc,
                                  "n":len(this_res), "base_rate":this_res["label"].mean()})
            print(f"{version} {pool_method} {label_col} F1: {f1:.2f} AUROC: {auroc:.2f} Accuracy: {acc:.2f}")
modelwise_res = pd.DataFrame(modelwise_res)
modelwise_res.to_csv(f"{repodir}/results/scanez_EML_modelwise_results.csv", index=False)

# %% t test comparing predicitons between models
comps = [
    ("ptEZ_mean", "ptEZftEML_mean"),
    ("ptEZ_mean", "ptEML_mean"),
    ("ptEZftEML_mean", "ptEML_mean"),
    ("ptEZ_final_pos", "ptEZftEML_final_pos"),
    ("ptEZ_final_pos", "ptEML_final_pos"),
    ("ptEZftEML_final_pos", "ptEML_final_pos"),
    ("ptEZ_masked_mean", "ptEZftEML_masked_mean"),
    ("ptEZ_masked_mean", "ptEML_masked_mean"),
    ("ptEZftEML_masked_mean", "ptEML_masked_mean"),
]
comp_res = []
for comp in comps:
    version1, pool_method1 = comp[0].split("_",1)
    version2, pool_method2 = comp[1].split("_",1)
    model1 = res_all[(res_all["version"]==version1) & (res_all["pool_method"]==pool_method1)]
    model2 = res_all[(res_all["version"]==version2) & (res_all["pool_method"]==pool_method2)]
    diff1 = model1["diff"].to_numpy()
    diff2 = model2["diff"].to_numpy()
    t, p = ttest_rel(diff1, diff2)
    print(f"{comp[0]} vs {comp[1]} t: {t:.2f} p: {p:.2f}")
    comp_res.append({"model1":comp[0], "model2":comp[1], "t":t, "p":p})
comp_res = pd.DataFrame(comp_res)
comp_res.to_csv(f"{repodir}/results/scanez_EML_model_comparisons.csv", index=False)
# %%
# check why res_all diff id the same for all models
res_all.groupby(["version","pool_method","label_col"])['diff'].mean()
# look at one id for all models
res_all[res_all["id"]==list(set(res_all["id"]))[20]]

# %%
