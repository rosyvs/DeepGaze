import numpy as np
import pandas as pd
from pathlib import Path

def get_stats(dir, filenames=[],flag=-180):
    if len(filenames) > 0:
        dfs=[pd.read_csv(f, usecols=['XAvg','YAvg']) for f in Path(dir).glob("*.csv") if f.stem in filenames]
    else:
        dfs = [pd.read_csv(f, usecols=['XAvg','YAvg']) for f in Path(dir).glob("*.csv")]
    combined_df = pd.concat(dfs)
    combined_df.replace(to_replace=flag, value=np.NaN, inplace=True)
    mean = combined_df.mean()
    std = combined_df.std()
    print(f'mean {mean} std {std}')
    return mean, std

if __name__ == "__main__":
    print(get_stats("./data/processed/fixation"))