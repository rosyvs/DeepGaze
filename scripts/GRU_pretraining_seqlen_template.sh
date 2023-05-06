#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00 
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=multitask-GRU-pretraining-fold-%j
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/GRUpre.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roso8920@colorado.edu

echo "running multitask_informer_pretraining.py"


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
version="fold${1}seqlen${3}"
name="GRU_pre_seqlen_seed${2}"
split_filepath="./data_splits/4fold_participant/seed${2}.yml"
seqlen=${3}
echo $name
echo $version
echo $split_filepath
echo $seqlen

python3 eyemind/experiments/multitask_GRU_pretraining.py -c experiment_configs/cluster/multitask_GRU_pretraining.yml --fold_number $1 --seed_everything $2 --split_filepath ${split_filepath} --trainer.logger.init_args.name ${name} --trainer.logger.init_args.version ${version}  --data.sequence_length ${seqlen} --model.sequence_length ${seqlen}
