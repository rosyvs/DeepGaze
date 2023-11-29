#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --job-name=informer-fixation-alpine
#SBATCH --cpus-per-task=1
#SBATCH --output=informer-fixation-alpine-exp.%j.out

module load cudnn/8.1
module load cuda/11.3


# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
conda activate dg
# pip install .
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python3 eyemind/experiments/informer_experiment.py fit -c configs/informer_test_config.yml