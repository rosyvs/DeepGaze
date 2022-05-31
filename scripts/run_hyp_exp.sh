#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --partition=sgpu
#SBATCH --ntasks=1 # always use 1 
#SBATCH --job-name=fix-hyp-exp
#SBATCH --output=fix-hyp-exp.%j.out

module load cuda/11.2
module load cudnn/8.1_for_cuda_11.2

# Run script
source ~/.bashrc
cd /projects/$USER/eyemind
pip install .
python3 eyemind/experiments/cross_val_fixation.py --data_folderpath ../fixation --log_dirpath lightning_logs --experiment_logging_folder seqlen_hiddendim_testing --sequence_length 100 250 500 --gru_hidden_size 128 256 --gpus 1 --max_epochs 8 --log_every_n_steps 1000
