#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00 
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=multitask-informer-pretraining-fold-%j
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/multitask-informer-pretraining-exp-fold.%j.out
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
pip install -e . # surely this doesnt need to be done as conda env already has eyemind?? 
echo "Fold: $1"
echo "Seed: $2"
version="fold${1}"
name="informer_pretraining_seed${2}"
split_filepath="./data_splits/4fold_participant/seed${2}.yml"
resume_dir=${3}
echo $name
echo $version
echo $split_filepath
echo $resume_dir

if [ -z "$resume_dir"]
then
  python3 eyemind/experiments/multitask_informer_pretraining.py -c experiment_configs/cluster/multitask_informer_pretraining_folds.yml --fold_number $1 --seed_everything $2 --split_filepath ${split_filepath} --trainer.logger.init_args.name ${name} --trainer.logger.init_args.version ${version}
else
  python3 eyemind/experiments/multitask_informer_pretraining.py -c experiment_configs/cluster/multitask_informer_pretraining_folds.yml --fold_number $1 --seed_everything $2 --split_filepath ${split_filepath} --trainer.logger.init_args.name ${name} --trainer.logger.init_args.version ${version} --trainer.resume_from_checkpoint ${resume_dir}
fi
