#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --job-name=multitask-test-alpine
#SBATCH --cpus-per-task=8
#SBATCH --output=multitask-test-alpine-exp.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roso8920@colorado.edu

module load cudnn/8.1
module load cuda/11.3


# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze/
conda activate dg
# pip install .
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python3 eyemind/experiments/multitask_experiment.py fit -c configs/encdec_multitask_config.yml --trainer.max_epochs 5