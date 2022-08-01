import numpy as np
import pandas as pd
from pathlib import Path

def get_stats(dir, flag=-180):
    dfs = [pd.read_csv(f, usecols=['XAvg','YAvg']) for f in Path(dir).glob("*.csv")]
    combined_df = pd.concat(dfs)
    combined_df.replace(to_replace=flag, value=np.NaN, inplace=True)
    mean = combined_df.mean()
    std = combined_df.std()
    print(combined_df.min(axis=0))
    return mean, std

if __name__ == "__main__":
    print(get_stats("/Users/rickgentry/emotive_lab/eyemind/data/processed/fixation"))