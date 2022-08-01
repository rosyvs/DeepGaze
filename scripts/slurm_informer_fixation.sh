#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=informer-fix
#SBATCH --cpus-per-task=1
#SBATCH --output=informer-fix-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/eyemind
conda activate eyemind
pip install .
python3 eyemind/experiments/informer_experiment.py fit -c experiment_configs/informer_test_config.yml --trainer.max_epochs 10