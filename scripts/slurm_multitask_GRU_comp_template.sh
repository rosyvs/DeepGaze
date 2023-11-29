#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=GRUcomp-fold-%j
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/GRUcomp.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roso8920@colorado.edu

echo "running multitask_GRU_comp.py"

module purge
module load cudnn/8.2
module load cuda/11.4

# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze

conda init bash
conda activate dg
# pip install -e . # surely this doesnt need to be done as conda env already has eyemind?? 
echo "Fold: $1"
echo "Seed: $2"
echo "Encoder Checkpoint: $3"
echo "Label Column: $4"
name="GRU_${4}"
version="fold${1}"
split_filepath="./data_splits/4fold_participant/seed${2}.yml"

echo $name
echo $version
echo $split_filepath


python3 eyemind/experiments/multitask_GRU_comp.py -c configs/cluster/multitask_GRU_comp.yml --fold_number $1 --split_filepath ${split_filepath} --model.encoder_ckpt $3 --data.label_col $4 --trainer.logger.init_args.name ${name} --trainer.logger.init_args.version ${version}