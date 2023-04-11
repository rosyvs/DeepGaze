import argparse
import subprocess
from pathlib import Path

# This python script automates running a slurm bash script for each fold. 
# Run this with -s slurm_multitask_informer_comp_template.sh

def main(args):
    for i in range(args.num_folds):
        ckpt_dirpath = Path(args.base_dir, f"fold{i}", "checkpoints")
        if args.last_ckpt:
            ckpt_path = str(next(ckpt_dirpath.glob('last*.ckpt')))
        else:
            ckpt_path = str(next(ckpt_dirpath.glob('epoch*.ckpt')))
        cmd = f"sbatch {args.slurm_script} {i} {ckpt_path} {args.label_col} {args.resume_ckpt}"
        print(cmd)
        cmd_list = cmd.split(" ")
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_script", required=True, help="Path to the tempalate slurm script to run")
    parser.add_argument("-f", "--num_folds", required=True, type=int, help="Number of folds to run")
    parser.add_argument("-d", "--base_dir", required=True, help="encoder_ckpt base directory path")
    parser.add_argument("--resume_ckpt", type=str, default="", help="checkpoint to resume training from")

    parser.add_argument("--last_ckpt", action='store_true', help="If you want to use the last checkpoint instead of the best saved one")
    parser.add_argument("-l", "--label_col", default="Rote_X" ,help="Comprehension label column name")
    args = parser.parse_args()
    main(args)