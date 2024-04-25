import argparse
import subprocess
from pathlib import Path
import glob
import os

# This python script automates running a slurm bash script for each fold. 
# e.g.
# python scripts/run_slurm_multitask_informer_pretrain.py 
# --folds 0 1 2 3\
# --config configs/cluster/new_multitask_informer_pretraining_folds.yml \
# --resume_ckpt lightning_logs/informer_pretraining_seed21/ 
# --last_ckpt

def main(args):
    if not args.config:
        config='configs/cluster/new_multitask_informer_pretraining_folds.yml'
    else:
        config=args.config
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
        
        print(f"Fold: {i}")
        version = f"fold{i}"

        # use subprocess to call the python script
        if ckpt_path:
            cmd = f"python eyemind/experiments/new_multitask_informer_pretraining.py -c {config} --fold_number {i} --trainer.logger.init_args.version {version} --trainer.resume_from_checkpoint {ckpt_path}"
        else:
            cmd = f"python eyemind/experiments/new_multitask_informer_pretraining.py -c {config} --fold_number {i} --trainer.logger.init_args.version {version}"
        print(cmd)
        cmd_list = cmd.split(" ")
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folds", required=True, type=int, nargs='*', help="list of fold numbers to run (e.g. 0 1 2 3)")
    parser.add_argument("-c", "--config", type=str, default="", help="Path to the yaml of hyperparameters (should end _folds.yml)")
    parser.add_argument("-r","--resume_dir", type=str, default="", help="base dir containing checkpoint to resume training from")
    parser.add_argument("-l","--last_ckpt", action='store_true', help="If you want to use the last checkpoint instead of the best saved one")

    args = parser.parse_args()
    main(args)    

    # note that last_ckpt might not work if resuming after an OOM crash caused during checkpoint saving