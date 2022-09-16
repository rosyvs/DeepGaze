#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=fixation-test-summit
#SBATCH --cpus-per-task=4
#SBATCH --output=fixation-test-summit-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/eyemind
conda activate eyemind
pip install .
python3 eyemind/experiments/fixation_experiment.py fit -c experiment_configs/fix_testing_config.yml 
