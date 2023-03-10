#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=comprehension-tune
#SBATCH --cpus-per-task=4
#SBATCH --output=comprehension-tune-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
conda activate eyemind
pip install .
python3 eyemind/experiments/comprehension_nested_cv.py -c experiment_configs/comprehension_nestedcv_config.yml --gpus_per_trial 1 --exp_name comprehension_tuning
