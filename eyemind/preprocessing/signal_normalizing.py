from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm

def convert_to_sample_rate(df,current,target):
    step_size = round(current/target)
    num_rows=len(df)
    sampled_df = df.iloc[np.arange(0,num_rows,step=step_size)]
    return sampled_df

# They seem to only use the y_axis to compute the visual angle, but this seems like it could be a problem?
def get_pixels_per_degree(screen_res,screen_size,subject_dist):
    x_pixels_per_degree = screen_res[0]/math.degrees(2*np.arctan2(screen_size[0],(2*subject_dist)))
    y_pixels_per_degree = screen_res[1]/math.degrees(2*np.arctan2(screen_size[1],(2*subject_dist)))
    return x_pixels_per_degree,y_pixels_per_degree

def get_screen_limits(screen_res,pixels_per_deg):
    x_degrees = screen_res[0] / pixels_per_deg[0] / 2
    y_degrees = screen_res[1] / pixels_per_deg[1] / 2
    return x_degrees,y_degrees

def convert_to_angle(df,screen_center,pixel_degrees):
    data = df.copy()
    data['XAvg'] = (df['XAvg'] - screen_center[0]) / pixel_degrees[0]
    data['YAvg'] = (df['YAvg'] - screen_center[1]) / pixel_degrees[1]
    return data

def get_time_signal(df): # restart timestamps at each event change
    min_tsample_df = df.groupby('event')[['tSample']].min().rename(columns={"tSample":"min_tSample"})
    # min_tsample_df = min_tsample_df.filter(items=['event','min_tSample']).reset_index()
    res_df = df.merge(min_tsample_df,on='event')
    res_df['t'] = res_df['tSample'] - res_df['min_tSample']
    return res_df


def write_file_event(df,output_path):
    for event in df.event.unique():
        temp_df = df.loc[df['event']==event]  
        name = f'{temp_df["ParticipantID"].iloc[0]}-{event}.csv'
        temp_df.to_csv(Path(output_path, name),index=False)

def plot_scanpath(df,event=None,exclude=None):
    if event:
        temp_df = df.loc[df['event']==event]
    else: 
        temp_df=df
    if exclude: # filter out flag vals
        temp_df=temp_df[(temp_df["XAvg"]!=exclude) & (temp_df["YAvg"]!=exclude)]
    # plt.plot(temp_df["t"], temp_df["XAvg"], label="x")
    # plt.plot(temp_df["t"], temp_df["YAvg"], label="y")
    plt.plot(temp_df["XAvg"], temp_df["YAvg"])

    plt.legend()
    name = f'{temp_df["ParticipantID"].iloc[0]} {temp_df["event"].iloc[0]}'
    plt.title(name)
    plt.show() 


def preprocess_data(
    raw_data_path, 
    output_folder, 
    screen_res=(1920,1080), 
    screen_size=(525.78,297.18),
    target_frequency=60, 
    current_frequency=1000, 
    subject_dist=989,
    NA_FLAG=-180, 
    off_screen_buf=5, 
    label_cols=[],
    debug=False):

    for file_path in tqdm(Path(raw_data_path).glob('*.csv')):
        print(f"Processing File Path: {file_path}")
        start = time.time()
        cols = ['ParticipantID','XAvg','YAvg','event','tSample'] + label_cols
        df = pd.read_csv(file_path,usecols=cols)
        sampled_df = convert_to_sample_rate(df,current_frequency,target_frequency)
        pixels_per_deg = get_pixels_per_degree(screen_res,screen_size,subject_dist)
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

        # Select Columns needed
        res_df = res_df.filter(items=['ParticipantID','XAvg','YAvg','event','t', 'tSample'])

        if debug:
            for event in df.event.unique():
                plot_scanpath(res_df,event, exclude=NA_FLAG)
        # Write files
        else:
            write_file_event(res_df,output_folder)
        print(f"Processed {file_path} in {(time.time() - start):.1f} seconds")



def main():
    # Stats of setup (EML)
    screen_res = (1920,1080)
    screen_size = (525.78,297.18)
    subject_dist = 989
    off_screen_buf=5
    print("Calling main")
    data_folder = "./data/"
    preprocess_data(raw_data_path=os.path.join(data_folder,"raw/sample"), 
    output_folder=Pos.path.join(data_folder,"processed/output"))

if __name__ == "__main__":
    main()