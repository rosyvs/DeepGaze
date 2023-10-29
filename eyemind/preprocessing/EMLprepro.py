#%%
import eyemind
repodir = os.path.dirname(os.path.dirname(eyemind.__file__))

from eyemind.preprocessing.fixations import preprocess_fixation, create_fixation_df
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

data_folder = Path("./data/")
output_folder = os.path.join(repodir,"data/EML/gaze")
os.makedirs(output_folder, exist_ok=True)


#%% preprocess gaze data

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

#%% add labels to gaze data


#%% add reading speed labels to gaze data