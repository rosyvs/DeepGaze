import argparse
import subprocess
from pathlib import Path

# This python script automates running a slurm bash script for each fold. 
# Run this with -s slurm_multitask_informer_comp_template.sh

def main(args):
    for i in args.folds:
        ckpt_dirpath = Path(args.encoder_dir, f"fold{i}", "checkpoints")
        if args.last_ckpt:
            ckpt_path = str(next(ckpt_dirpath.glob('last*.ckpt')))
        else:
            ckpt_path = str(next(ckpt_dirpath.glob('epoch*.ckpt')))
        cmd = f"{args.slurm_script} {i} {ckpt_path} {args.label_col} {args.resume_dir}"
        print(cmd)
        cmd_list = cmd.split(" ")
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_script", required=True, help="Path to the slurm script to run")
    parser.add_argument("-f", "--folds", required=True, type=int, nargs='*', help="list of fold numbers to run (e.g. 0 1 2 3)")
    parser.add_argument("-d", "--encoder_dir", required=True, help="encoder_ckpt base directory path")
    # parser.add_argument("--resume_dir", type=str, default="", help="checkpoint to resume training from")

    parser.add_argument("--last_ckpt", action='store_true', help="For loading pretrained encoder, use the last checkpoint instead of the best saved one")
    parser.add_argument("-l", "--label_col", default="Rote_X" ,help="Comprehension label column name")
    args = parser.parse_args()
    main(args)

        # note that last_ckpt might not work if resuming after an OOM crash caused during checkpoint saving