#%%
import eyemind
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))
from eyemind.models.loss import get_class_weights
from eyemind.preprocessing.fixations import preprocess_fixation, create_fixation_df, label_fixations, label_gaze_timeseries
from eyemind.preprocessing.signal_normalizing import interleave_samples, get_pixels_per_degree, get_screen_limits, get_pixels_per_degree, get_time_signal, convert_to_angle, write_file_event,plot_scanpath
from pathlib import Path
import os
import pandas as pd
import re
from eyemind.preprocessing.pygaze_detectors import saccade_detection, fixation_detection, blink_detection, faster_saccade_detection
pd.options.mode.chained_assignment = None  # default='warn'
%reload_ext autoreload
%autoreload 2
from tqdm import tqdm
import time



# NOTE: this version does not downsample as before, but takes interleaved sequences, to make 
# more data!

#%% setup paths and params
screen_res = (1920,1080)
screen_size = (525.78,297.18)
subject_dist = 989
sample_every= 16
current_frequency=1000
target_frequency=1000.0/sample_every

NA_FLAG=-180
off_screen_buf=5
label_cols=[]
debug=False
raw_data_path='/Users/roso8920/Dropbox (Emotive Computing)/EyeMindLink/GuojingData/sample'
repodir = '/Users/roso8920/Dropbox (Emotive Computing)/EyeMindLink/Processed/Gaze/Timeseries/DeepGaze/'
output_folder = os.path.join(repodir,"EML","gaze_interleaved16")
os.makedirs(output_folder, exist_ok=True)
#%%
PREPRO_DONE=True
if not PREPRO_DONE:
    for file_path in tqdm(Path(raw_data_path).glob('*.csv'), total=len(list(Path(raw_data_path).glob('*.csv')))):
        print(f"Processing File Path: {file_path}")
        start = time.time()
        cols = ['ParticipantID','XAvg','YAvg','event','tSample'] + label_cols
        df = pd.read_csv(file_path,usecols=cols)
        sampled_dfs = interleave_samples(df,sample_every)
        pixels_per_deg = get_pixels_per_degree(screen_res,screen_size,subject_dist)
        pID = df['ParticipantID'].unique()
        if len(pID)>1:
            print(f"Multiple participant IDs in file {file_path}, that is not expected")
            break
        else:
            pID=pID[0]
        for i,sampled_df in enumerate(sampled_dfs):
            sampled_df = convert_to_angle(sampled_df,screen_center=(screen_res[0]//2,screen_res[1]//2),pixel_degrees=pixels_per_deg)
            # Set off screen gaze to NA_FLAG = -180 
            x_lim, y_lim = get_screen_limits(screen_res,pixels_per_deg)
            sampled_df.loc[sampled_df['XAvg'] < -x_lim - off_screen_buf, 'XAvg'] = NA_FLAG
            sampled_df.loc[sampled_df['XAvg'] > x_lim + off_screen_buf, 'XAvg'] = NA_FLAG
            sampled_df.loc[sampled_df['YAvg'] < -y_lim - off_screen_buf, 'YAvg'] = NA_FLAG
            sampled_df.loc[sampled_df['YAvg'] > y_lim + off_screen_buf, 'YAvg'] = NA_FLAG
            # Set null vals (blinks) to NA_FLAG = -180
            sampled_df.loc[sampled_df['XAvg'].isna(),'XAvg'] = NA_FLAG
            sampled_df.loc[sampled_df['YAvg'].isna(),'YAvg'] = NA_FLAG

            # Get time signal
            res_df = get_time_signal(sampled_df)
            res_df['t'] = res_df['t']+i # offset by i samples - needed for accurate fixation labeling later

            # Select Columns needed
            res_df = res_df.filter(items=['ParticipantID','XAvg','YAvg','event','t', 'tSample'])

            if debug:
                for event in res_df.event.unique():
                    plot_scanpath(res_df, event, exclude=NA_FLAG)
            # Write files
            else:
                for event in res_df.event.unique():
                    temp_df = res_df.loc[res_df['event']==event]  
                    name = f'{pID}-{event}-i{i}.csv'
                    temp_df.to_csv(Path(output_folder, name), index=False)
        print(f"Processed {file_path} in {(time.time() - start):.1f} seconds")


#%% load labels for fixations
regressions_path = os.path.join(repodir,"EML","regressions_info","regressions_reading_pages.csv")
reg_df = pd.read_csv(regressions_path)
reg_df['filename'] = reg_df['ParticipantID']+ '-' + reg_df['identifier']

#%% apply fixation binary label
# gaze_path = os.path.join(repodir,'EML/gaze_interleaved16')

# # FIXATION LABEL (uses regression df)
# labelled_folder = os.path.join(repodir,"EML/gaze+fix_interleaved16")
# os.makedirs(labelled_folder, exist_ok=True)
# # apply_label_df(regressions_path, gaze_path, labelled_folder, label_name='fixation_label', onset_col='CURRENT_FIX_START',offset_col='CURRENT_FIX_END', time_col='t' )
# label_name='fixation_label'
# onset_col='CURRENT_FIX_START'
# offset_col='CURRENT_FIX_END'
# time_col='t' # TODO: but this is tSample by default in the function...
# label_file=regressions_path
# label_df = pd.read_csv(label_file)
# label_df["filename_base"] = label_df.apply(lambda row: f"{row['ParticipantID']}-{row['identifier']}",axis=1)
# label_df[label_name]=1
# grouped = label_df.groupby("filename_base")
# fix_stats=[]
# for filename_base, df_group in tqdm(grouped, total=len(grouped)):
#     for i in range(sample_every):
#         filename = f"{filename_base}-i{i}.csv"
#         try:
#             gaze_df = pd.read_csv(os.path.join(gaze_path,filename))
#         except Exception as e:
#             # print(f"Couldn't read file: {os.path.join(gaze_path,filename)} because of {e}")
#             continue
#         labeled_df = label_gaze_timeseries( gaze_df, df_group,label_name,onset_col, offset_col,time_col)
#         if labeled_df is not None:
#             labeled_df[label_name] = labeled_df[label_name].fillna(0) # saccade if not fixation
#             labeled_df.to_csv(Path(labelled_folder, filename),index=False)
#             fix_stats.append(labeled_df[label_name].value_counts().sort_index().rename(filename.replace('.csv','')))


# fix_stats=pd.DataFrame(fix_stats)
# fix_stats.to_csv(Path(labelled_folder + '_counts.csv'))
# classes=list(fix_stats.columns)
# fix_stats['n']=fix_stats.sum(axis=1)
# print('fixation class %:')
# print('macroaverage:')
# print(100*fix_stats[classes].sum()/fix_stats['n'].sum())
# print('microaverage:')
# print(100*(fix_stats[classes]/fix_stats['n']).mean())

#%% apply fixation/regression 3-class label
# 3-class label (2=regression, 1= other fixation, 0=not a fixation)
gaze_path = os.path.join(repodir,"EML/gaze_interleaved16") # apply labels to the df with binary fix labels already
labelled_folder = os.path.join(repodir,"EML/gaze+fix+reg_interleaved16")
os.makedirs(labelled_folder, exist_ok=True)
onset_col='CURRENT_FIX_START'
offset_col='CURRENT_FIX_END'
time_col='t' 

label_file=regressions_path

label_df = pd.read_csv(label_file)
label_df["filename_base"] = label_df.apply(lambda row: f"{row['ParticipantID']}-{row['identifier']}",axis=1)
label_df['regression_label']=1+label_df['LOCAL_REGRESSION']# after adding 1, 2 if regression, 1 if on-text, NaN if fixation not in IA
label_df['regression_label']=label_df['regression_label'].fillna(1) # all non-regression fixations now have label 1
grouped = label_df.groupby("filename_base")
reg_stats=[]
for filename_base, df_group in tqdm(grouped, total=len(grouped)):
    for i in range(sample_every):
        filename = f"{filename_base}-i{i}.csv"
        try:
            gaze_df = pd.read_csv(os.path.join(gaze_path,filename))
        except Exception as e:
            # print(f"Couldn't read file: {os.path.join(gaze_path,filename)} because of {e}")
            continue
        labeled_df = label_gaze_timeseries( gaze_df, df_group, 'regression_label',onset_col, offset_col,time_col)

        if labeled_df is not None:
            labeled_df['regression_label'] = labeled_df['regression_label'].fillna(0).astype(int) # third category is non-fixations
            labeled_df['fixation_label'] = (labeled_df['regression_label']>0.5).astype(int) # binary fixation_label
            # labeled_df[label_name] = labeled_df[label_name].replace({0:'fix_other',1:'fix_reg',2:'not_fix'})
            labeled_df.to_csv(Path(labelled_folder, filename),index=False)
            reg_stats.append(labeled_df['regression_label'].value_counts().sort_index().rename(filename.replace('.csv','')))

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

#%% select instances and get summary stats
data_path =  os.path.join(repodir,'EML/EML1_pageLevel_500+_matchEDMinstances.csv')
instances = pd.read_csv(data_path)
# add reading speed labels to gaze data
text=pd.read_csv(os.path.join(repodir,'EML/texts-char-word-counts.csv')).rename(columns={'text':'Text','pageNum':'PageNum'})
instances=instances.merge(text,how='left',on=['Text','PageNum'])
instances['readWPM']=instances['wordCount']/instances['readtime']*60

instances.rename(columns={'filename':'filename_base'},inplace=True)

reg_stats = pd.read_csv(os.path.join(repodir,"EML/gaze+fix+reg_interleaved16_counts.csv")).rename(columns={'Unnamed: 0':'filename'})
classes=list(reg_stats.drop('filename', axis=1).columns)
reg_stats['n']=reg_stats.sum(axis=1,numeric_only=True)
# add filename_base column to reg stats and merge with instances to get interleaved instances
reg_stats['filename_base'] = reg_stats['filename'].apply(lambda x: re.sub(r'-i\d+','',x))
instances_interleaved=instances.merge(reg_stats,how='left', on='filename_base') # instances interleaved

weights=get_class_weights(instances_interleaved[classes].sum()/instances_interleaved['n'].sum())
print(f'regression class ratio in selected instances: {list(round(instances_interleaved[classes].sum()/instances_interleaved["n"].sum(),3))}')
print(f'regression class weights: {weights}')

# Fixation class weights
fix_instances_interleaved=instances_interleaved.copy()
fix_instances_interleaved['fix0'] = fix_instances_interleaved['0']
fix_instances_interleaved['fix1'] = fix_instances_interleaved['1'] + fix_instances_interleaved['2']
weights=get_class_weights(fix_instances_interleaved[['fix0','fix1']].sum()/fix_instances_interleaved['n'].sum())
print(f'fixation class ratio in selected instances: {list(round(fix_instances_interleaved[["fix0","fix1"]].sum()/fix_instances_interleaved["n"].sum(),3))}')
print(f'fixation class weights: {weights}')

instances_interleaved.to_csv(os.path.join(repodir,'EML/EML1_pageLevel_500+_matchEDMinstances_interleaved16.csv'),index=False)

#%% compute gaze coordinate mean and sd
from eyemind.preprocessing.standardizing import get_stats
data_path =  os.path.join(repodir,'EML/EML1_pageLevel_500+_matchEDMinstances_interleaved16.csv')
instances = pd.read_csv(data_path)
mean,std=get_stats(os.path.join(repodir,"EML/gaze+fix+reg_interleaved16"), filenames=list(instances['filename']))
print(f"mean: {mean}, std: {std}")

# %% TODO: instance lsit not liimited to EDMinstances
