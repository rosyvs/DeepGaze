import argparse
import subprocess
from pathlib import Path

# This python script automates running a slurm bash script for each fold. 
# Run this with -s slurm_multitask_informer_pretraining_template.sh

#e.g.
# python scripts/run_slurm_seqlens_GRU_pretrain.py 
# -f 0 1 2 3 --seed 21 \
# -l 50 200 500

def main(args):
    for i in args.folds:
        for l in args.sequence_lens: 
            cmd = f"sbatch scripts/GRU_pretraining_seqlen_template.sh {i} {args.seed} {l}"
            print(cmd)
            cmd_list = cmd.split(" ")
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folds", required=True, type=int, nargs='*', help="list of fold numbers to run (e.g. 0 1 2 3)")
    parser.add_argument("--seed", type=int, default=21, help="Seed for pytorch-lightning")
    parser.add_argument("-l","--sequence_lens", required=True, type=int, nargs='*', help="gaze sequence length (e.g. 50 200 500)")
    args = parser.parse_args()
    main(args)    

    # note that last_ckpt might not work if resuming after an OOM crash caused during checkpoint saving