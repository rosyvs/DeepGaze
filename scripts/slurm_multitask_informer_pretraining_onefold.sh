#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=multitask-informer-pretraining-fold-%j
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-informer-pretraining-exp-fold.%j.out

module load cudnn/8.1
module load cuda/11.3

# Run script
source /projects/$USER/.bashrc_alpine
cd /projects/$USER/eyemind
conda activate pya100
echo "Fold: $1"
echo "Seed: $2"
name="informer_pretraining_seed${2}"
version="fold${1}"
split_filepath="/projects/rige3027/eyemind/data_splits/4fold_participant/seed${2}.yml"
echo $name
echo $version
echo $split_filepath
python3 eyemind/experiments/multitask_informer_pretraining.py -c experiment_configs/cluster/multitask_informer_pretraining_folds.yml --fold_number $1 --seed_everything $2 --split_filepath ${split_filepath} --trainer.logger.init_args.name ${name} --trainer.logger.init_args.version ${version}