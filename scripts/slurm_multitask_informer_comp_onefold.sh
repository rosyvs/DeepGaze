#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=multitask-informer-comp-fold-%j
#SBATCH --cpus-per-task=4
#SBATCH --output=multitask-informer-comp-exp-fold.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roso8920@colorado.edu

echo "running multitask_informer_comp.py"

module purge
module load cudnn/8.1
module load cuda/11.3


# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
conda init bash
conda activate dg
echo "Fold: $1"
echo "Encoder Checkpoint: $2"
echo "Label Column: $3"
name="informer_${3}"
version="fold${1}"
split_filepath="./data_splits/4fold_participant/seed${2}.yml"
echo $name
echo $version
echo $split_filepath

python3 eyemind/experiments/multitask_informer_comp.py -c experiment_configs/cluster/multitask_informer_comp.yml --fold_number $1 --model.encoder_ckpt $2 --data.label_col $3 --trainer.logger.init_args.name ${name} --trainer.logger.init_args.version ${version}