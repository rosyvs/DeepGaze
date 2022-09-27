#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=multitask-test-alpine
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-test-alpine-exp.%j.out

module load cudnn/8.1
module load cuda/11.3


# Run script
source /projects/$USER/.bashrc_alpine
cd /projects/$USER/eyemind
conda activate pya100
pip install .
pip install torch==1.12.1+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python3 eyemind/experiments/informer_experiment.py fit -c experiment_configs/informer_test_config.yml