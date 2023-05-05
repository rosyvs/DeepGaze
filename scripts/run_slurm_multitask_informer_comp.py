import argparse
import subprocess
from pathlib import Path

# This python script automates running a slurm bash script for each fold. 
# Run this with -s slurm_multitask_informer_comp_template.sh

#e.g.
# python scripts/run_slurm_multitask_informer_comp.py \
# -s scripts/slurm_multitask_informer_comp_template.sh \
# -f 0 1 2 3 --seed 21 \
# --encoder_dir lightning_logs/informer_pretraining_seed21/ \
# --label_col Rote_X

def main(args):
    if args.label_col=="ALL":
        label_list = ['Rote_X', 'Rote_Y', "Rote_Z",
          'Inference_X', "Inference_Y","Inference_Z",
          "Deep_X", "Deep_Z",
          "MW"]
    else:
        label_list=[args.label_col]  
    for l in label_list:
        for i in args.folds:
            ckpt_dirpath = Path(args.encoder_dir, f"fold{i}", "checkpoints")
            if args.last_ckpt:
                ckpt_path = str(next(ckpt_dirpath.glob('last*.ckpt')))
            else: # get most recent checkpoint
                # files=ckpt_dirpath.glob('epoch*.ckpt')
                files=ckpt_dirpath.glob('*.ckpt')
                latest_file = max(list(files), key=lambda item: item.stat().st_ctime)
                ckpt_path = str(latest_file)    
            cmd = f"sbatch {args.slurm_script} {i} {args.seed} {ckpt_path} {l}"
            print(cmd)
            cmd_list = cmd.split(" ")
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            print("stdout:", result.stdout)
            # print("stderr:", result.stderr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_script", required=True, help="Path to the tempalate slurm script to run")
    parser.add_argument("-f", "--folds", required=True, type=int, nargs='*', help="list of fold numbers to run (e.g. 0 1 2 3)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for pytorch-lightning, should match that of encoder")
    parser.add_argument("-d", "--encoder_dir", required=True, help="encoder_ckpt base directory path")
    parser.add_argument("--last_ckpt", action='store_true', help="If you want to use the last checkpoint instead of the best saved one")
    parser.add_argument("-l", "--label_col", default="Rote_X" ,help="Comprehension label column name. ALL for all")
    args = parser.parse_args()
    main(args)