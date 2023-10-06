# convert BEdata to format used by EML
#%%
import eyemind
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))

from eyemind.preprocessing.fixations import preprocess_fixation, create_fixation_df
from eyemind.preprocessing.signal_normalizing import *
from pathlib import Path
import os
import pandas as pd
import re
pd.options.mode.chained_assignment = None  # default='warn'
%reload_ext autoreload
%autoreload 2
# Experimental setup:
# Tobii TX300 !! 300 Hz sampling rate, this is consistent w timestamps being in microseconds

screen_res = (1920,1080)
screen_size = (509, 286) #(mm?) source: http://screen-size.info/ 23 inch 16x9
subject_dist = 650# (mm?? )
screen_center = (screen_res[0]//2,screen_res[1]//2)
current_frequency = 300
target_frequency= 60
off_screen_buf=5
# Flag for off screen or NaN values
NA_FLAG = -180 # Flag for off screen gaze. 
INVALID_REJECT_THRESHOLD = 0.2 # TODO: reject any timeseries consisting of at least this much invalid (either x or y missing ) data
DEBUG=False
DO_PREPRO = False
# load sample-level data
raw_data_path = '/Users/roso8920/Dropbox (Emotive Computing)/BE15/Data/Data Checking/GazeLogs/'

output_folder = os.path.join(repodir,"data/BE15/gaze")
os.makedirs(output_folder, exist_ok=True)

if DO_PREPRO:
    for file_path in Path(raw_data_path).glob('*.txt'):
        print(f"Processing File Path: {file_path}")
        start = time.time()
        cols = ['ParticipantID',
        'LeftGazePointX',	
        'LeftGazePointY',	
        'RightGazePointX',	
        'RightGazePointY',
        'CurrentItemName',
        'CurrentItemPosition'
        ,'EyeTrackerTimeStamp']
        df = pd.read_csv(file_path,sep='\t',usecols=cols).rename(columns={'EyeTrackerTimeStamp':'tSample'})
        pID=df['ParticipantID'][0]
        if 'TEST' in pID: # skip test
            continue
        # compute XAvg and YAvg
        df['XAvg']=df[[   'LeftGazePointX',	'RightGazePointX']].mean(axis=1)
        df['YAvg']=df[[   'LeftGazePointY',	'RightGazePointY']].mean(axis=1)
        df.drop(columns=['LeftGazePointX',	
        'LeftGazePointY',	
        'RightGazePointX',	
        'RightGazePointY'], inplace=True)
        df['event'] = df['CurrentItemName'].str.replace('C:/EmotiveComputingLab/BE15/Materials/','').str.replace('.txt','')+\
            df['CurrentItemPosition'].astype(str).str.pad(width=3, fillchar='0')
        # timestamps of EML, and BE fixatoin labels are in ms , gaze is in us
        df['tSample'] = round(df['tSample']/1000).astype(int)


        sampled_df = convert_to_sample_rate(df,current_frequency,target_frequency) # just taking every nth
        pixels_per_deg = get_pixels_per_degree(screen_res,screen_size,subject_dist)
        sampled_df = convert_to_angle(sampled_df,(screen_res[0]//2,screen_res[1]//2),pixels_per_deg)
        # Set off screen gaze to NA_FLAG = -180 
        x_lim, y_lim = get_screen_limits(screen_res,pixels_per_deg) # pretty sure this should be divided by 2... deg from center
        sampled_df.loc[sampled_df['XAvg'] < -x_lim - off_screen_buf, 'XAvg'] = NA_FLAG
        sampled_df.loc[sampled_df['XAvg'] > x_lim + off_screen_buf, 'XAvg'] = NA_FLAG
        sampled_df.loc[sampled_df['YAvg'] < -y_lim - off_screen_buf, 'YAvg'] = NA_FLAG
        sampled_df.loc[sampled_df['YAvg'] > y_lim + off_screen_buf, 'YAvg'] = NA_FLAG
        # Set null vals (blinks) to NA_FLAG = -180
        sampled_df.loc[sampled_df['XAvg'].isna(),'XAvg'] = NA_FLAG
        sampled_df.loc[sampled_df['YAvg'].isna(),'YAvg'] = NA_FLAG

        # Get time signal
        res_df = get_time_signal(sampled_df)

        # keep only MainText gaze 
        res_df=res_df[res_df['event'].str.contains('MainText')]

        # Select Columns needed
        res_df = res_df.filter(items=['ParticipantID','XAvg','YAvg','event','t', 'tSample']).reset_index(drop=True)

        if DEBUG:
            for event in res_df.event.unique():
                plot_scanpath(res_df,event, exclude=NA_FLAG)
        # Write files
        else:
            write_file_event(res_df,output_folder)
        print(f"Processed {file_path} in {time.time() - start} seconds")


# %% compute fixations from raw data
from eyemind.preprocessing.pygaze_detectors import saccade_detection, fixation_detection, blink_detection
labelled_folder = os.path.join(repodir,"data/BE15/gaze+fix+sac")
os.makedirs(labelled_folder, exist_ok=True)

for i,file_path in enumerate(Path(output_folder).glob('*.csv')):
    df=pd.read_csv(file_path)
    fname=file_path.stem
    if 'TEST' in fname:
        continue
    df=pd.read_csv(file_path)

    # although these functions are in pixels our data is now in degrees, so thresholds are in degrees
    # "max" in params is misleading, these are the minimum vals to be cosidered a sacccade, or max to not be considered a saccade
    # minlen is in ms not distance! #TODO: implement this myself
    #  
    Ssac, Esac = saccade_detection(
        x=df['XAvg'].to_numpy(), 
        y=df['YAvg'].to_numpy(), 
        time=df['t'].to_numpy(), 
        missing=NA_FLAG,
        minlen=5, 
        maxvel=30, 
        maxacc=9500
        )
    if Esac: # if any detected:
        # use this to make sac / fix labels
        # Esac	-	list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
        sacdf = pd.DataFrame({'tStart':[sac[0] for sac in Esac], 'tEnd':[sac[1] for sac in Esac]})
        # merge onto df to get tSamples column
        sacdf=sacdf.merge(df[['t','tSample']], how="left", left_on='tStart', right_on='t').rename(columns={'tSample':'StartTimestamp'}).drop(columns='t')
        sacdf=sacdf.merge(df[['t','tSample']], how="left", left_on='tEnd', right_on='t').rename(columns={'tSample':'EndTimestamp'}).drop(columns='t')
        sacdf['SaccadeIndex']=sacdf.index
        # 
        # fix_labels_expanded = create_fixation_df(fix_labels)
        sacdf['t'] = sacdf.apply(lambda row: list(range(int(row['tStart']), int(row['tEnd']))), axis=1)
        sac_labels_expanded = sacdf.explode("t")
        sac_labels_expanded["saccade_label"] = 1
        sac_labels_expanded = sac_labels_expanded.drop(['tStart','tEnd'], axis=1).reset_index(drop=True)
        labelled_df = df.merge(sac_labels_expanded, how='left', on='t')
        labelled_df['saccade_label'] = labelled_df['saccade_label'].fillna(0)

        # fixations as inverse of saccades minus flag values
        labelled_df['fixation_label'] = 1-labelled_df['saccade_label']
        labelled_df['missing'] = (labelled_df['XAvg']==NA_FLAG) | (labelled_df['XAvg']==NA_FLAG)
        labelled_df['fixation_label'].iloc[labelled_df['missing']]=0

        # stats 
        print(f"fixation %: {100*labelled_df['fixation_label'].sum()/len(labelled_df):.2f}")
        print(f"saccade %: {100*labelled_df['saccade_label'].sum()/len(labelled_df):.2f}")
        print(f"missing %: {100*labelled_df['missing'].sum()/len(labelled_df):.2f}")
        labelled_df.to_csv(Path(labelled_folder, f'{fname}.csv'),index=False)
    else: # none detected
        print(f'no saccades detected for {fname}')

# %% load fixation labels: TODO: use recomputed
fixation_path = '/Users/roso8920/Dropbox (Emotive Computing)/BE15/Data/Fixations/BE15-fixations_Avg-Filtered.txt'
fix_labels_all = pd.read_csv(fixation_path, sep='\t', \
    usecols=['ParticipantID', 'FixationIndex','StartTimestamp','EndTimestamp',]).rename(
        columns={'StartTimestamp':'tStart', 'EndTimestamp':'tEnd'})
fix_labels_all['ParticipantID'] = fix_labels_all['ParticipantID'].astype(str)
fix_labels_all['tStart'] = fix_labels_all['tStart'].astype(int)
fix_labels_all['tEnd'] = fix_labels_all['tEnd'].astype(int)

labelled_folder = os.path.join(repodir,"data/BE15/gaze+fix")
os.makedirs(labelled_folder, exist_ok=True)

for i,file_path in enumerate(Path(output_folder).glob('*.csv')):
    df=pd.read_csv(file_path)
    fname=file_path.stem
    if 'TEST' in fname:
        continue
    # these were made 1 df per pID-event combination
    pID=df['ParticipantID'][0]
    begin=df['tSample'][0]
    end=df['tSample'].iloc[-1]

    if pID not in list(fix_labels_all['ParticipantID']):
        print(f'no fixation label exists for participant {pID}')
        continue
    fix_labels = fix_labels_all[(fix_labels_all['ParticipantID'].astype(str)==pID)]
    fix_labels = fix_labels[(fix_labels['tEnd'] >= begin) & (fix_labels['tStart' ]<=end)]
    if fix_labels.empty:
        print(f'no fixation label exists for {fname}')
        continue

    # fix_labels_expanded = create_fixation_df(fix_labels)
    fdf = fix_labels[['tStart', 'tEnd']]#.reset_index(drop=True)
    fdf['tSample'] = fdf.apply(lambda row: list(range(int(row['tStart']), int(row['tEnd']))), axis=1)
    fix_labels_expanded = fdf.explode("tSample")
    fix_labels_expanded["fixation_label"] = 1
    fix_labels_expanded = fix_labels_expanded.drop(['tStart','tEnd'], axis=1).reset_index(drop=True)
    labelled_df = df.merge(fix_labels_expanded, how='left', on='tSample')
    labelled_df['fixation_label'] = labelled_df['fixation_label'].fillna(0)

    # if i==0:
    #     break

    if labelled_df is not None:
        labelled_df.to_csv(Path(labelled_folder, f'{fname}.csv'),index=False)


# %%Make "labelfile" which has 1 row per page and lists all instances to be used

# columns needed: filename, sequence_length? 

fnames=[]
seqlens=[]
pIDs=[]
pagenums=[]
labelled_folder = os.path.join(repodir,"data/BE15/gaze+fix")

for i,file_path in enumerate(Path(labelled_folder).glob('*.csv')):
    df=pd.read_csv(file_path)
    fname=file_path.stem # fname contains event name
    seqlen=len(df)
    pID=df['ParticipantID'][0]
    event=df['event'][0]

    PageNum=int(re.search(r'MainText(\d*)', fname).group(1))

    if 'TEST' in fname:
        continue

    if seqlen:
        fnames.append(fname)
        seqlens.append(seqlen)
        pIDs.append(pID)
        pagenums.append(PageNum)

instances = pd.DataFrame(
    {'filename':fnames,
    'sequence_length':seqlens,
    'ParticipantID':pIDs,
    'PageNum':pagenums,
    'event':event}
).sort_values(by='filename')
print(f'{len(instances)} instances')
instances=instances.reset_index(drop=True)
instances.to_csv(os.path.join(repodir,"data/BE15/", 'BE15_instances.csv'))


# filter instances to not include reread
instances_g = instances.groupby(["ParticipantID","PageNum"]).filter(lambda x: len(x) <= 1) # drop any with group longer than 1
print(f'{len(instances_g)} instances kept after removing pages with rereads')
instances_g=instances_g.reset_index(drop=True)
instances_g.to_csv(os.path.join(repodir,"data/BE15/", 'BE15_instances_read1x.csv'))

# %%
