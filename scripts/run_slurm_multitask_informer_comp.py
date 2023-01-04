import argparse
import subprocess

def main(args):
    cmd = f"sbatch {args.slurm_script} {args.fold_num} {args.ckpt_path} {args.label_col}"
    cmd_list = cmd.split(" ")
    result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_script", required=True, help="Path to the slurm script to run")
    parser.add_argument("-f", "--fold_num", required=True, help="What fold number to run")
    parser.add_argument("-p", "--ckpt_path", required=True, help="encoder_ckpt path")
    parser.add_argument("-l", "--label_col", default="Rote_X" ,help="Comprehension label column name")
    args = parser.parse_args()
    main(args)