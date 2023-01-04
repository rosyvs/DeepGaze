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


###
Focus on different tasks to see if it does well on any of them

Temporal vs spatial features:

What features can you predict for spatial features? Use a CNN to predict without the temporal aspect 

