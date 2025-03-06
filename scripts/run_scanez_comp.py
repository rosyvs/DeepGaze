import argparse
import subprocess
from pathlib import Path

# This python script automates running a slurm bash script for each fold. 

#e.g.
# python scripts/run_limu_comp.py \
# -c configs/2024/lvcmp/classifier_limubert_125.yaml \
# -f 0 1 2 3 \
# --label_col ALL
# # %%
# ScanEZ embeddings (I extracted train and val) 
#     Train: new_split{1,2,3,4}_embeddings.npy
#      Val:  new_split{1,2,3,4}_embeddings_val.npy

# W/o Pretraining (e.g., only trained on human data)  
#      Train: new_Only_split{12,3,4}_embeddings.npy
#      Val: new_Only_split{1,2,3,4}_embeddings_val.npy

# W/o Finetuning (e.g., only trained on synthetic data) 
#      Train: newEZ_Reader_Only_split{1,2,3,4}_embeddings.npy 
#      Val: newEZ_Reader_Only_split{12,3,4}_embeddings_val.npy

# The data inside eml_structured_21 and eml_unstructured_21 are the npy files used to train/finetune the LIMU model 

# params for making paths:
# split
# VAL (bool). valstr = "_val" if VAL else ""
# part "train" or "val". Should be "train" when VAL is False, otherwise "val"
# version_long 

path_templates = {
    'ptEZftEML': 'data/ekta_embeddings/new_split{split}_embeddings{valstr}.npy/rosie_{part}_{split}_with_embedding_usingnew_e_pretrain_200_FT_EML_sentid_split_fold{split}.npy',
    'ptEZ': 'data/ekta_embeddings/newEZ_Reader_Only_split{split}_embeddings{valstr}.npy/rosie_{part}_{split}_with_embedding_usingnew_ez_pretrain_200.npy',
    'ptEML': 'data/ekta_embeddings/new_Only_split{split}_embeddings{valstr}.npy/rosie_{part}_{split}_with_embedding_usingeml_only_200_sentid_fold{split}.npy'
}
# Define the possible values for pooling and version
pooling_methods = ["mean", "final_pos", "masked_mean"]
versions = ["ptEZftEML", "ptEZ", "ptEML"]
labels = ['SVT','Rote_X', 'Rote_Y', "Rote_Z",
          'Inference_X', "Inference_Y","Inference_Z",
          "Deep_X", "Deep_Z",
          "Rote_D","Inference_D",
          "MW"]
def main(args):

    v = args.version
    p = args.pooling
    if v == 'ALL':  
        ver_list = versions
    else:
        ver_list = [v]
    if p == 'ALL':
        pool_list = pooling_methods
    else:
        pool_list = [p]

    if not args.config:
        config='configs/2025/cluster/train_classifier_scanez.yaml'
    else:
        config=args.config
    if args.label_col=="ALL":
        label_list = labels
    else:
        label_list=[args.label_col]  
    for v in ver_list:
        for p in pool_list:
            for l in label_list:
                for i in args.folds:
                    print(f"Fold: {i}")
                    split=i+1
                    train_path = path_templates[v].format(split=split, valstr="", part="train")
                    val_path = path_templates[v].format(split=split, valstr="_val", part="val")
                    # print(f"Train path: {train_path}")
                    # print(f"Val path: {val_path}")
                    if not Path(train_path).exists():
                        print(f"Train Path {train_path} does not exist")
                        continue
                    if not Path(val_path).exists():
                        print(f"Val Path {val_path} does not exist")
                        continue
                    name=f"{l}_scanez_{v}_{p}"
                    version = f"fold{i}"
                    cmd = f'python eyemind/experiments/limu_comp.py -c {config} --data.label_col {l} --data.pool_method {p} --data.train_data_path {train_path} --data.val_data_path {val_path} --trainer.logger.init_args.name {name} --trainer.logger.init_args.version {version}'
                    print(cmd)
                    cmd_list = cmd.split(" ")
                    result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
                    print("stdout:", result.stdout)
                    print("stderr:", result.stderr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folds", required=True, type=int, nargs='*', help="list of fold numbers to run (e.g. 0 1 2 3)")
    parser.add_argument("-c", "--config", type=str, default="", help="Path to the yaml of hyperparameters (should end _folds.yml)")
    parser.add_argument("-y", "--label_col", default="ALL" ,help="Comprehension label column name. ALL for all")
    parser.add_argument("-v", "--version", default="ALL", help="Version string describing embeddings source. Options: ptEZftEML, ptEZ, ptEML. ALL for all")
    parser.add_argument("-p", "--pooling", default="ALL", help="Pooling function for embeddings (mean, final_pos, masked_mean). ALL for all")
    args = parser.parse_args()
    main(args)