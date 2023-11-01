import argparse
import subprocess
from pathlib import Path
import glob
import os
# This python script automates running a slurm bash script for each fold. 
# Run this with a fold-level template in command line argument --slurm_script
# e.g. slurm_multitask_informer_pretraining_template.sh 
# slurm_new_multitask_informer_pretraining_template.sh

#e.g.
# python scripts/run_slurm_multitask_informer_pretrain.py \
# -s scripts/slurm_multitask_informer_pretraining_template.sh \
# -f 0 1 2 3 --seed 21 \
# --resume_ckpt lightning_logs/informer_pretraining_seed21/

def main(args):
    if not args.hparams:
        hparams='experiment_configs/cluster/new_multitask_informer_pretraining_folds.yml'
    else:
        hparams=args.hparams
    for i in args.folds:
        if args.resume_dir:
            ckpt_dirpath = Path(args.resume_dir, f"fold{i}", "checkpoints")
            if args.last_ckpt:
                ckpt_path = str(next(ckpt_dirpath.glob('last*.ckpt')))
            else: # get most recent checkpoint
                files=list(ckpt_dirpath.glob('*.ckpt'))
                if len(files)>0:
                    latest_file = max(files, key=lambda item: item.stat().st_ctime)
                    ckpt_path = str(latest_file)    
                else:
                    print(f'no existing ckpt in resume_dir for fold {i}')
                    ckpt_path = "" 
        else:
            ckpt_path = "" 
        cmd = f"sbatch {args.slurm_script} {i} {args.seed} {ckpt_path} {hparams}"
        print(cmd)
        cmd_list = cmd.split(" ")
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_script", required=True, help="Path to the template slurm script to run")
    parser.add_argument("-f", "--folds", required=True, type=int, nargs='*', help="list of fold numbers to run (e.g. 0 1 2 3)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for pytorch-lightning")
    parser.add_argument("--resume_dir", type=str, default="", help="base dir containing checkpoint to resume training from")
    parser.add_argument("--last_ckpt", action='store_true', help="If you want to use the last checkpoint instead of the best saved one")
    parser.add_argument("-p", "--hparams", type=str, default="", help="Path to the yaml of hyperparameters (should end _folds.yml)")

    args = parser.parse_args()
    main(args)    

    # note that last_ckpt might not work if resuming after an OOM crash caused during checkpoint saving