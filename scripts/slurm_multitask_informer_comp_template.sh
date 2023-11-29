#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=comp.%j
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/comp.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roso8920@colorado.edu

echo "running multitask_informer_comp.py"

module purge
module load cudnn/8.2
module load cuda/11.4

# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze

conda init bash
conda activate dg
# pip install -e . # surely this doesnt need to be done as conda env already has eyemind?? 
config=${2}

echo "Fold: $1"
echo $config
echo "Encoder Checkpoint: $3"
echo "Label Column: $4"
version="fold${1}"

echo $name
echo $version
echo $split_filepath


python3 eyemind/experiments/multitask_informer_comp.py -c ${config} --fold_number $1 --model.encoder_ckpt $3 --data.label_col $4 --trainer.logger.init_args.version ${version}