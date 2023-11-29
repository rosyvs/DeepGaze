#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=multitask-informer-pretraining-folds
#SBATCH --cpus-per-task=1
#SBATCH --output=multitask-informer-pretraining-exp-folds.%j.out

module load cudnn/8.1
module load cuda/11.3
module load parallel


# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
conda activate dg
# pip install .
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
#parallel -P $SLURM_NODES srun  -n 1 --exclusive python3 eyemind/experiments/multitask_informer_pretraining.py ::: {0..3}
srun  -n 1 -c 1 --exclusive python3 eyemind/experiments/multitask_informer_pretraining.py -c configs/multitask_informer_pretraining_folds.yml --fold_number 0 &
srun  -n 1 -c 1 --exclusive python3 eyemind/experiments/multitask_informer_pretraining.py -c configs/multitask_informer_pretraining_folds.yml --fold_number 1 &
srun  -n 1 -c 1 --exclusive python3 eyemind/experiments/multitask_informer_pretraining.py -c configs/multitask_informer_pretraining_folds.yml --fold_number 2 &
srun  -n 1 -c 1 --exclusive python3 eyemind/experiments/multitask_informer_pretraining.py -c configs/multitask_informer_pretraining_folds.yml --fold_number 3 &
wait