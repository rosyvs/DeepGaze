#!/bin/bash

#SBATCH --time=12:00:00 
#SBATCH --partition=aa100
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --job-name=scanez
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/scanez_comp.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=roso8920@colorado.edu
module purge
module load cudnn/8.1
module load cuda/11.3


# Run script
source ~/.bashrc
cd /projects/$USER/DeepGaze
conda activate dg
# pip install .
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

python scripts/run_scanez_comp.py -f 0 1 2 3 -v ptEZftEML -p mean -c configs/2025/cluster/train_classifier_scanez.yaml
python scripts/run_scanez_comp.py -f 0 1 2 3 -v ptEZ -p mean -c configs/2025/cluster/train_classifier_scanez.yaml
python scripts/run_scanez_comp.py -f 0 1 2 3 -v ptEML -p mean -c configs/2025/cluster/train_classifier_scanez.yaml

