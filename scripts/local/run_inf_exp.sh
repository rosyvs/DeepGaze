#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=fix-hyp-exp
#SBATCH --output=fix-hyp-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
# pip install .
python3 eyemind/experiments/inference_experiment.py fit -c experiment_configs/inf_cluster_config.yml
