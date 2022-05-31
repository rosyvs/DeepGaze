# eyemind


## Installation
```
pip install .
```
To see logs install tensorboard:
```
pip install tensorboard
```


## Cluster

### Setup
First setup as for common models redesign: https://github.com/emotive-computing/common-models-redesign 


Conda environment
```
conda create -n eyemind python=3.9
conda activate eyemind
```

Install code
```
git clone git@github.com:r3g2/eyemind.git
cd eyemind
pip install .
```


Download data from dropbox: https://www.dropbox.com/work/EyeMindLink/Processed/Rick_DeepLearning

Copy processed data to project directory. 
```
scp -r <path-to-folder> <username>@login.rc.colorado edu:<target-path>    # using a login node
```

