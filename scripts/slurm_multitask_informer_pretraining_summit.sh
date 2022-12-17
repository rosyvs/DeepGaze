#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=multitask-informer-pretraining
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-informer-pretraining-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/eyemind
conda activate eyemind
pip install .
python3 eyemind/experiments/multitask_informer_pretraining.py -c experiment_configs/multitask_informer_pretraining.yml