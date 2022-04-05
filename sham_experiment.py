from collections import OrderedDict
from random import random
from numpy import pad
from sklearn.ensemble import RandomForestClassifier
from torch import nn
import torch
import torch.nn.functional as F
import os 
from skorch.callbacks import Freezer
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path("../OBF").resolve()))
from obf.model import ae
from obf.model import creator

from preprocessing.run_pytorch_model import get_subject_groups, process_sham_data
from commonmodels2.models.model import PyTorchModel
from commonmodels2.stages.evaluation_stage import EvaluationStage, SupervisedEvaluationContext
from commonmodels2.stages.cross_validation import GenerateCVFoldsStage, NestedCrossValidationStage, CrossValidationStage, PyTorchNestedSupervisedCVContext, SupervisedCVContext
from commonmodels2.stages.load_data import CSVDataLoaderStage, ObjectDataLoaderStage
from commonmodels2.stages.pipeline import Pipeline
from commonmodels2.stages.preprocessing import ImputerPreprocessingStage, FeatureScalerPreprocessingStage, EncoderPreprocessingStage


pre_train_dir = str(Path("../OBF/pre_weights/sample_weights").resolve())
# Hyperparams
batch_size = 4
learning_rate = 0.001
epochs = 100
report_interval = 10
hidden_layers=[256,512]
use_cuda = torch.cuda.is_available()
train_type = "freeze"

def pytorch_model_func(params):
    encoder = creator.load_encoder(pre_train_dir,use_cuda=use_cuda)
    model = creator.create_classifier_from_encoder(encoder,hidden_layers=hidden_layers,n_output=1,dropout=0.5)
    creator.print_models_info(['original encoder', "current model"], [encoder,model])
    return model

def transform_predict(predict_output):
    probs = F.sigmoid(predict_output,dim=1)
    labels = probs > 0.5
    return labels

def RunML():
    # Create PyTorchModel and set fixed parameters 
    random_seed = 22
    pym = PyTorchModel()
    pym.set_model_create_func(pytorch_model_func)
    pym.set_fit_params({"lr": learning_rate, "batch_size":batch_size , "train_split": None, "predict_nonlinearity": transform_predict, "max_epochs": 100, 'callbacks': [Freezer(["*rnn*","*cnn*"])]})
    pym.set_criterion_params({"criterion": "binary_crossentropy_with_logits"})

    data_df = process_sham_data(Path("/Users/rickgentry/emotive_lab/eyemind/data/preprocessed/shamnosham_output"))
    x_np = np.stack(np.array(data_df["scanpath"].values))
    # df = data_df.sample(frac=1).reset_index(drop=True)
    # groups = get_subject_groups(df)
    s0 = ObjectDataLoaderStage()
    s0.setDataObject(data_df)
    feat_cols = ['scanpath']
    label_cols = ['sham']

    s1 = EncoderPreprocessingStage(label_cols, 'labelencoder')

    s2 = GenerateCVFoldsStage(strategy='random_grouped', strategy_args={'seed': random_seed, 'num_folds': 4, 'group_by': "ParticipantID"})

    cv_context_pym = SupervisedCVContext()
    cv_context_pym.model = pym
    cv_context_pym.feature_cols = feat_cols
    cv_context_pym.y_label = label_cols
    
    # params = {"model": {"activation_fn": [nn.ReLU,nn.GELU]}, 
    #             "optimizer": {"optimizer_fn": ["adam","sgd"]},
    #             "loss": {"loss_fn": "categorical_crossentropy"},
    #             "fit": {"lr": 1e-3, "batch_size":32 , "train_split": None, "predict_nonlinearity": transform_predict, "max_epochs": 100}
    #         }
    # cv_context_pym.param_eval_func = 'accuracy'
    # cv_context_pym.param_eval_goal = 'max'
    # # init training context for tensorflow supervised model with nested CV params
    # cv_context_tfm = TensorFlowSupervisedTrainParamGridContext()
    # cv_context_tfm.model = tfm
    # cv_context_tfm.feature_cols = cols
    # cv_context_tfm.y_label = categorical_cols
    # cv_context_tfm.eval_funcs = 'categorical_crossentropy'
    # cv_context_tfm.param_grid = {'hidden_layer_size': [3,4,5,6]}
    # cv_context_tfm.param_eval_goal = 'min'
    # cv_context_tfm.optimizer = 'sgd'

    s3 = CrossValidationStage()
    s3.setCVContext(cv_context_pym)
    eval_context = SupervisedEvaluationContext()
    eval_context.y_label = label_cols
    eval_context.eval_funcs = ['accuracy']
    s3.setEvaluationContext(eval_context)


    p = Pipeline()
    p.addStage(s0)
    p.addStage(s1)
    p.addStage(s2)
    p.addStage(s3)
    p.run()

    cv_results = p.getDC().get_item('cv_results')

    for fold in cv_results.keys():
        print(cv_results[fold])

if __name__ == '__main__':
    RunML()
