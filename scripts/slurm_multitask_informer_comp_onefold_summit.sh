#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1
#SBATCH --job-name=multitask-informer-pretraining-fold-%j
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-informer-pretraining-fold-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
conda activate eyemind
pip install -e .
echo "Fold: $1"
echo "Encoder Checkpoint: $2"
echo "Label Column: $3"
python3 eyemind/experiments/multitask_informer_comp.py -c experiment_configs/cluster/multitask_informer_comp.yml --fold_number $1 --model.encoder_ckpt $2 --data.label_col $3