import argparse
import subprocess
from pathlib import Path

# This python script automates running a slurm bash script for each fold. 
# Run this with -s slurm_multitask_informer_comp_template.sh

#e.g.
# python scripts/run_slurm_multitask_informer_comp.py \
# -c configs/cluster/new_multitask_informer_comp.yml \
# -f 0 1 2 3 \
# --encoder_dir lightning_logs/new_multitsk_informer_pretraining/ \
# --label_col Rote_X

def main(args):
    if not args.config:
        config='configs/2024/lvcmp/classifier_limubert_125.yaml'
    else:
        config=args.config
    if args.label_col=="ALL":
        label_list = ['SVT','Rote_X', 'Rote_Y', "Rote_Z",
          'Inference_X', "Inference_Y","Inference_Z",
          "Deep_X", "Deep_Z",
          "Rote_D","Inference_D",
          "MW"]
    else:
        label_list=[args.label_col]  
    for l in label_list:
        for i in args.folds:
            print(f"Fold: {i}")
            train_path = f'./rosie_train_{i+1}_with_embedding_using_rosie_512_train_split_{i+1}.npy'
            val_path = f'./data/limu/embeds_splits_val_512/rosie_val_{i+1}_with_embedding_using_rosie_512_train_split_{i+1}.npy'
            name=f"{l}_limu"
            version = f"fold{i}"
            cmd = f'python eyemind/experiments/limu_comp.py -c {config} --data.label_col {l} --trainer.logger.init_args.name {name} --trainer.logger.init_args.version {version}'
            print(cmd)
            cmd_list = cmd.split(" ")
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            # print("stdout:", result.stdout)
            # print("stderr:", result.stderr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folds", required=True, type=int, nargs='*', help="list of fold numbers to run (e.g. 0 1 2 3)")
    parser.add_argument("-c", "--config", type=str, default="", help="Path to the yaml of hyperparameters (should end _folds.yml)")
    parser.add_argument("-y", "--label_col", default="ALL" ,help="Comprehension label column name. ALL for all")
    args = parser.parse_args()
    main(args)