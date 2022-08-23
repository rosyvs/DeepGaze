#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=multitask-test-summit
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-test-summit-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/eyemind
conda activate eyemind
pip install .
python3 eyemind/experiments/multitask_experiment.py fit -c experiment_configs/encdec_multitask_config.yml 
