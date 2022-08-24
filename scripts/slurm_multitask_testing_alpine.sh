#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=multitask-test-alpine
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-test-alpine-exp.%j.out

module load cuda/11.2
module load cudnn/8.1

# Run script
source /projects/$USER/.bashrc_alpine
cd /projects/$USER/eyemind
conda activate eyemind
pip install .[cuda112] --use-feature=in-tree-build
python3 eyemind/experiments/multitask_experiment.py fit -c experiment_configs/encdec_multitask_config.yml --trainer.max_epochs 5