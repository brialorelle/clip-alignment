# %% read subject id from english_only_subjects.txt
import os
from glob import glob

english_subjects = []
with open('english_subjects.txt', 'r') as f:
    for line in f:
        english_subjects.append(line.strip())
        

# %%
folder_path = "/data/yinzi/babyview_20240507/transcripts_multilingual-large-v3/Babyview_Main"
for subject in english_subjects:
    all_subject_csv = glob(os.path.join(folder_path, subject, "*.csv"))

# read all csv files, extract english only
import pandas as pd
for subject_csv in all_subject_csv:
    df = pd.read_csv(subject_csv)
    # df is not empty
    if not df.empty:
        if df['language'][0] != 'en':
            print(subject_csv)
            print("not english")
            continue
    


# %%
import re
mp3_folder = "/data/yinzi/babyview_20240507/audios/Babyview_Main"
all_audio_files = sorted(glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True))
en_audio_files = []
for audio_file in all_audio_files:
    file_name = os.path.basename(audio_file)
    file_name = re.sub(r"\.mp3$", "", file_name)
    try:
        # fix it because the partten is different for luna vidoes
        partten = re.findall(r'(.*)_GX', file_name)
        if len(partten) == 0: # new name partten
            partten = re.findall(r'(.*)_H', file_name)
        if len(partten) == 0:  # fallback partten
            partten = file_name.split("_")
        subject_id = partten[0]
    except:
        print(f"[WARNING]: Could not extract subject ID from {file_name}, skip")
        raise RuntimeError
    if subject_id in english_subjects:
        en_audio_files.append(audio_file)
