import argparse
import subprocess
from pathlib import Path


def main(args):
    for i in range(args.num_folds):
        if args.resume_ckpt:
            ckpt_dirpath = Path(args.resume_ckpt, f"fold{i}", "checkpoints")
            if args.last_ckpt:
                ckpt_path = str(next(ckpt_dirpath.glob('last*.ckpt')))
            else:
                ckpt_path = str(next(ckpt_dirpath.glob('epoch*.ckpt')))        
        else:
            ckpt_path = ""
        cmd = f"sbatch {args.slurm_script} {i} {args.seed} {ckpt_path}"
        print(cmd)
        cmd_list = cmd.split(" ")
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_script", required=True, help="Path to the slurm script to run")
    parser.add_argument("-f", "--num_folds", required=True, type=int, help="Number of folds to run")
    parser.add_argument("--seed", type=int, default=42, help="Seed for pytorch-lightning")
    parser.add_argument("--resume_ckpt", type=str, default="", help="base dir containing checkpoint to resume training from")
    parser.add_argument("--last_ckpt", action='store_true', help="If you want to use the last checkpoint instead of the best saved one")
    args = parser.parse_args()
    main(args)    

    # note that last_ckpt might not work if resuming after an OOM crash caused during checkpoint saving