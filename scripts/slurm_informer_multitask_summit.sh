#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=multitask-informer
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-informer-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/eyemind
conda activate eyemind
pip install .
python3 eyemind/experiments/informer_experiment.py fit -c experiment_configs/inf_multitask_config.yml