#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --job-name=multitask-informer
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-informer-exp.%j.out

module load cudnn/8.1
module load cuda/11.3


# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
conda activate dg
# pip install .
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python3 eyemind/experiments/informer_experiment.py fit -c experiment_configs/inf_multitask_config.yml