from pathlib import Path
import pandas as pd


def fixation_label_mapper(folder, files):
    labels = []
    for f in files:
        df = pd.read_csv(str(Path(folder,f).resolve()))
        label_array = df['fixation_label'].to_numpy(float)
        labels.append(label_array)
    return labels

def create_fixation_df(df):
    df = df[['tStart', 'tEnd']]
    df['tSample'] = df.apply(lambda row: list(range(row['tStart'], row['tEnd'])), axis=1)
    res_df = df.explode("tSample")
    res_df["fixation_label"] = 1
    res_df = res_df.drop(['tStart','tEnd'], axis=1).reset_index(drop=True)
    return res_df

def label_fixations(data_path, filename, label_df):
    try:
        df = pd.read_csv(Path(data_path, filename))
    except Exception as e:
        print(f"Couldn't read file: {str(Path(data_path, filename))} because of {e}")
        return None
    fixation_df = create_fixation_df(label_df)
    #print(fixation_df.head())
    res_df = df.merge(fixation_df, how='left', on='tSample')
    res_df['fixation_label'] = res_df['fixation_label'].fillna(0)
    return res_df

def preprocess_fixation(fixation_folder_path, full_data_path, output_path):
    fixation_folder_path = Path(fixation_folder_path)
    for file_path in fixation_folder_path.glob('*.csv'):
        df = pd.read_csv(file_path)
        df["filename"] = df.apply(lambda row: f"{row['ParticipantID']}-{row['event']}.csv",axis=1)
        grouped = df.groupby("filename")
        for group_name, df_group in grouped:
            labeled_df = label_fixations(full_data_path, group_name, df_group)
            if labeled_df is not None:
                labeled_df.to_csv(Path(output_path, group_name),index=False)

def main():
    #print(label_fixations("/Users/rickgentry/emotive_lab/eyemind/data/raw/sample", "EML1_003.csv", pd.read_csv("/Users/rickgentry/emotive_lab/eyemind/data/fixation/EML1_003.csv")).head())
    preprocess_fixation("/Users/rickgentry/emotive_lab/eyemind/data/fixation", "/Users/rickgentry/emotive_lab/eyemind/data/processed/output", "/Users/rickgentry/emotive_lab/eyemind/data/processed/fixation")
    

if __name__ =="__main__":
    main()