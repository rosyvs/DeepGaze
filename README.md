# eyemind


## Installation

### Local
1. Create conda environment with python >=3.9
```
conda create -n eyemind python=3.10
conda activate eyemind
```
2. Install code

```
git clone git@github.com:emotive-computing/EML-DeepLearning.git
cd eyemind
pip install .
```
To see logs install tensorboard:
```
pip install tensorboard
```

3. Make sure your project directory has a data/ folder and Download data from dropbox to that folder: https://www.dropbox.com/work/EyeMindLink/Processed/Rick_DeepLearning.

### Cluster

1. Setup as for common models redesign: https://github.com/emotive-computing/common-models-redesign 

2. Follow steps in Local to create conda environment and install code

3. For alpine gpus need to run special install

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

4. Copy data folder with processed data from local to cluster. 
```
scp -r <path-to-folder> <username>@login.rc.colorado edu:<target-path>
```

## Project Structure


```
   
   EYEMIND
   ├── data  ## Base directory for all data
   │   ├── processed ## Used for the data that has gone through preprocessing
   ├── data_splits ## Contains splits of data for different cross validation strategies
   │   └── 4fold_participant.yml ## 4 fold split of train, val by participant id
   ├── configs ## Lightning config files containing parameters for running experiments
   │   └── cluster ## For running on the cluster
   │   └── local ## For running on your local machine   
   ├── eyemind  ## Main python package
   │   └── analysis ## Utilities for visualizing outputs (fixations, etc...)
   │   └── dataloading ## Pytorch datasets, dataloaders, and pytorch-lightning datamodules
   │   └── experiments ## Entrypoints that have clis that run training, validation, testing
   │   └── models ## Contains pytorch-lightning LightningModules that have code for running the RNN and informer models (encoderdecoders for multitask and classifiers for comprehension)
   │   └── obf ## From Oculomotor Behavior Framework paper (not used for much)
   │   └── preprocessing ## Python scripts that output processed data. Shouldn't need to be run again.
   │   └── trainer ## Specialized loops for pytorch-lightning (not used in usual process)
   ├── scripts ## Bash and python scripts to run slurm jobs on the cluster   
   ├── .gitignore
   ├── LICENSE.txt
   ├── MANIFEST.in
   ├── README.md
   ├── setup.cfg
   ├── pyproject.toml
   └── 

```

## Experiments

### Running an experiment locally

1. Use a current experiment script or create a new python script in eyemind/experiments that will act as your cli

2. Update the config file which specifies hyperparameters for splitting the data, running the training, the model, and the data loading

3. If you want to override specific hyperparameters in the config file you can specify them when running the python script. (This is useful when running on cluster with slurm)

Example: To run pretraining with the informer model use: 
```
python eyemind/experiments/multitask_informer_pretraining.py -c configs/local/multitask_informer_pretraining.yml --num_folds 4 --seed_everything 25 --split_filepath ./data_splits/4fold_participant/seed25.yml
```

This runs the script multitask_informer_pretraining.yml using the config file multitask_informer_pretraining.yml and overrides a few of the parameters. This allows you not to have to create a ton of config files when trying to run different fold splits.  The python script will split the data into folds, save the splits to a file,instantiate the trainer, model, and datamodule then run the training process.

If you want to see the options that the script can be run with you can run:

```
python eyemind/experiments/multitask_informer_pretraining.py --help 
```

### Running an experiment on the Alpine Cluster

1. Use a current slurm job script or create a new one under the scripts/ folder

Example:
 To run pretraining with the informer model do the following:

```
python scripts/run_slurm_multitask_informer_pretrain.py -s scripts/slurm_multitask_informer_pretraining_template.sh -f 4 --seed 22
```
 This will run the slurm job script 4 times, one for each fold, "slurm_multitask_informer_pretraining_template.sh" with the random seed equal to 22. 

 The slurm job script ends up running the python script shown in the local experiment (multitask_informer_pretraining.py):
 ```
 python3 eyemind/experiments/multitask_informer_pretraining.py -c configs/cluster/multitask_informer_pretraining_folds.yml --fold_number $1 --seed_everything $2 --split_filepath ${split_filepath} --trainer.logger.init_args.name ${name} --trainer.logger.init_args.version ${version}
 ```

 It passes the fold number it is currently running and overrides the directories for the logs and splits.