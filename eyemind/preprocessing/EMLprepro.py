#%%
import eyemind
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))
from eyemind.models.loss import get_class_weights
from eyemind.preprocessing.fixations import preprocess_fixation, create_fixation_df, label_fixations, label_gaze_timeseries
from eyemind.preprocessing.signal_normalizing import *
from pathlib import Path
import os
import pandas as pd
import re
from eyemind.preprocessing.pygaze_detectors import saccade_detection, fixation_detection, blink_detection, faster_saccade_detection
pd.options.mode.chained_assignment = None  # default='warn'
%reload_ext autoreload
%autoreload 2


#%% setup paths and params
screen_res = (1920,1080)
screen_size = (525.78,297.18)
subject_dist = 989
target_frequency= 60
current_frequency=1000

output_folder = os.path.join(repodir,"data/EML/gaze")
os.makedirs(output_folder, exist_ok=True)


#%% preprocess gaze data (downsample, normalize etc)

preprocess_data(
    raw_data_path='/Users/roso8920/Dropbox (Emotive Computing)/EyeMindLink/GuojingData/sample', 
    output_folder=output_folder,
    screen_res=screen_res, 
    screen_size=screen_size,
    target_frequency=target_frequency, 
    current_frequency=current_frequency, 
    subject_dist=subject_dist,
    NA_FLAG=-180, 
    off_screen_buf=5, 
    label_cols=[],
    debug=False
    )

#%% load labels for fixations
regressions_path = Path(f"{repodir}/data/regressions_info/regressions_reading_pages.csv")
reg_df = pd.read_csv(regressions_path)
reg_df['filename'] = reg_df['ParticipantID']+ '-' + reg_df['identifier']
gaze_path = os.path.join(repodir,'data/EML/gaze')

#%% apply fixation binary label
# FIXATION LABEL (uses regression df)
labelled_folder = os.path.join(repodir,"data/EML/gaze+fix")
os.makedirs(labelled_folder, exist_ok=True)
# apply_label_df(regressions_path, gaze_path, labelled_folder, label_name='fixation_label', onset_col='CURRENT_FIX_START',offset_col='CURRENT_FIX_END', time_col='t' )
label_name='fixation_label'
onset_col='CURRENT_FIX_START'
offset_col='CURRENT_FIX_END'
time_col='t' 
label_file=regressions_path
label_df = pd.read_csv(label_file)
label_df["filename"] = label_df.apply(lambda row: f"{row['ParticipantID']}-{row['identifier']}.csv",axis=1)
label_df[label_name]=1
grouped = label_df.groupby("filename")
fix_stats=[]
for filename, df_group in grouped:
    try:
        gaze_df = pd.read_csv(os.path.join(gaze_path,filename))
    except Exception as e:
        # print(f"Couldn't read file: {os.path.join(gaze_path,filename)} because of {e}")
        continue
    labeled_df = label_gaze_timeseries( gaze_df, df_group,label_name,onset_col, offset_col,time_col)
    if labeled_df is not None:
        labeled_df[label_name] = labeled_df[label_name].fillna(0) # saccade if not fixation
        labeled_df.to_csv(Path(labelled_folder, filename),index=False)
        fix_stats.append(labeled_df[label_name].value_counts().sort_index().rename(filename.replace('.csv','')))


fix_stats=pd.DataFrame(fix_stats)
fix_stats.to_csv(Path(labelled_folder + '_counts.csv'))
classes=list(fix_stats.columns)
fix_stats['n']=fix_stats.sum(axis=1)
print('fixation class %:')
print('macroaverage:')
print(100*fix_stats[classes].sum()/fix_stats['n'].sum())
print('microaverage:')
print(100*(fix_stats[classes]/fix_stats['n']).mean())

#%% apply fixation/regression 3-class label
# 3-class label (2=regression, 1= other fixation, 0=not a fixation)
gaze_path = os.path.join(repodir,"data/EML/gaze+fix") # apply labels to the df with binary fix labels already

labelled_folder = os.path.join(repodir,"data/EML/gaze+fix+reg")
os.makedirs(labelled_folder, exist_ok=True)
label_name='regression_label'
onset_col='CURRENT_FIX_START'
offset_col='CURRENT_FIX_END'
time_col='t' 

label_file=regressions_path

label_df = pd.read_csv(label_file)
label_df["filename"] = label_df.apply(lambda row: f"{row['ParticipantID']}-{row['identifier']}.csv",axis=1)
label_df[label_name]=1+label_df['LOCAL_REGRESSION']# after adding 1, 2 if regression, 1 if on-text, NaN if fixation not in IA
label_df[label_name]=label_df[label_name].fillna(1) # all non-regression fixations now have label 1
grouped = label_df.groupby("filename")
reg_stats=[]
for filename, df_group in grouped:
    try:
        gaze_df = pd.read_csv(os.path.join(gaze_path,filename))
    except Exception as e:
        # print(f"Couldn't read file: {os.path.join(gaze_path,filename)} because of {e}")
        continue
    labeled_df = label_gaze_timeseries( gaze_df, df_group, label_name,onset_col, offset_col,time_col)
    if labeled_df is not None:
        labeled_df[label_name] = labeled_df[label_name].fillna(0).astype(int) # third category is non-fixations
        # labeled_df[label_name] = labeled_df[label_name].replace({0:'fix_other',1:'fix_reg',2:'not_fix'})
        labeled_df.to_csv(Path(labelled_folder, filename),index=False)
        reg_stats.append(labeled_df[label_name].value_counts().sort_index().rename(filename.replace('.csv','')))

print('regression class %')
reg_stats=pd.DataFrame(reg_stats)
reg_stats.to_csv(Path(labelled_folder + '_counts.csv'))

classes=list(reg_stats.columns)
reg_stats['n']=reg_stats.sum(axis=1)
print('overall regression class %:')
# print('macroaverage:')
print(round(100*reg_stats[classes].sum()/reg_stats['n'].sum(),1))
# print('microaverage:')
# print(100*(reg_stats[classes]/reg_stats['n']).mean())

#%% select instances ang get summary stats
data_path =  os.path.join(repodir,'data/processed/EML1_pageLevel_500+_matchEDMinstances.csv')
instances = pd.read_csv(data_path)

# filter stats by instances
fix_stats = pd.read_csv(os.path.join(repodir,"data/EML/gaze+fix_counts.csv"))
fix_stats.rename(columns={'Unnamed: 0':'filename'}, inplace=True)
classes=list(fix_stats.drop('filename', axis=1).columns)
fix_stats['n']=fix_stats.sum(axis=1,numeric_only=True)
fix_stats=instances[['filename']].merge(fix_stats,how='left')
weights=get_class_weights(fix_stats[classes].sum()/fix_stats['n'].sum())
print(f'fixation ratio in selected instances: {list(round(fix_stats[classes].sum()/fix_stats["n"].sum(),3))}')
print(f'fixation class weights: {weights}')

reg_stats = pd.read_csv(os.path.join(repodir,"data/EML/gaze+fix+reg_counts.csv")).rename(columns={'Unnamed: 0':'filename'})
classes=list(reg_stats.drop('filename', axis=1).columns)
reg_stats['n']=reg_stats.sum(axis=1,numeric_only=True)
reg_stats=instances[['filename']].merge(reg_stats,how='left')

weights=get_class_weights(reg_stats[classes].sum()/reg_stats['n'].sum())
print(f'regression class ratio in selected instances: {list(round(reg_stats[classes].sum()/reg_stats["n"].sum(),3))}')
print(f'regression class weights: {weights}')


#%% add reading speed labels to gaze data
df=pd.read_csv(os.path.join(repodir,'data/EML/EML1_pageLevel_500+_matchEDMinstances.csv'))
text=pd.read_csv(os.path.join(repodir,'data/EML/texts-char-word-counts.csv')).rename(columns={'text':'Text','pageNum':'PageNum'})
df=df.merge(text,how='left',on=['Text','PageNum'])
df.columns
df['readWPM']=df['wordCount']/df['readtime']*60
df.to_csv('/Users/roso8920/Dropbox (Emotive Computing)/EML Rosy/DeepGaze/data/EML/EML1_pageLevel_500+_matchEDMinstances.csv')

#%% compute gaze coordinate mean and sd
from eyemind.preprocessing.standardizing import get_stats
data_path =  os.path.join(repodir,'data/processed/EML1_pageLevel_500+_matchEDMinstances.csv')
instances = pd.read_csv(data_path)
mean,std=get_stats(os.path.join(repodir,"data/EML/gaze+fix+reg"), filenames=list(instances['filename']))
# %%
