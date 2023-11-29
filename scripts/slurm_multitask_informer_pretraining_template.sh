#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00 
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --job-name=multitask-informer-pretraining-fold-%j
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/pretraining.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roso8920@colorado.edu

echo "running new_multitask_informer_pretraining.py with new pretraining tasks"


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
version="fold${1}"
config=${2}
resume_dir=${3}
echo $version
echo $resume_dir
echo $config

if [ -z "$resume_dir"]
then
  python3 eyemind/experiments/new_multitask_informer_pretraining.py -c ${config} --fold_number $1 --trainer.logger.init_args.version ${version}
else
  python3 eyemind/experiments/new_multitask_informer_pretraining.py -c ${config} --fold_number $1 --trainer.logger.init_args.version ${version} --trainer.resume_from_checkpoint ${resume_dir}
fi
