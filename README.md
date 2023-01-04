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
   ├── experiment_configs ## Lightning config files containing parameters for running experiments
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

