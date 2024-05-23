#%%
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

# Paths setup
babyview_video_folder = "/data/yinzi/babyview_20240507/Babyview_Main"
output_root_dir = "/data/yinzi/babyview_20240507"
all_subject_number_list = sorted(os.listdir(babyview_video_folder))
# exclude pilot subjects
exclude_subjects = ['Bria_Long', 'Erica_Yoon']
all_subject_number_list = [subject for subject in all_subject_number_list if subject not in exclude_subjects]
total_number_of_videos = sum([len(glob(os.path.join(babyview_video_folder, subject, "*.MP4"))) for subject in all_subject_number_list])

all_results = []  # This will store results from all subjects

merged_subjects_df = pd.DataFrame()

for subject in tqdm(all_subject_number_list, desc="Processing Subjects"):
    all_mp4_files = glob(os.path.join(babyview_video_folder, subject, "*.MP4"))
    subject_df = pd.DataFrame()
    for mp4_full_path in tqdm(all_mp4_files, desc="Processing Videos"):
        video_file_name = os.path.splitext(os.path.basename(mp4_full_path))[0]
        date = "".join(video_file_name.split("_")[2:])
        output_csv_dir = os.path.join(output_root_dir, "all_clip_results_large_v3", subject, "all_result_csv_files", video_file_name, 'clip_final_results.csv')
        
        # Checking if the result file exists
        if not os.path.exists(output_csv_dir):
            print(f"Missing data for {video_file_name}")
            continue
        
        single_video_df = pd.read_csv(output_csv_dir)
        single_video_df = single_video_df.dropna(subset=['dot_product'])  # skip NaN values
        single_video_df['utterance_no'] = single_video_df['utterance_no'].apply(lambda x: f"{video_file_name}_{x}")  # rename the utterance_no
        single_video_df['video_name'] = video_file_name  # add video name
        single_video_df['date'] = date
        # merge all the dataframes
        subject_df = pd.concat([subject_df, single_video_df])
    merged_subjects_df = pd.concat([merged_subjects_df, subject_df])
        
# %%
merged_subjects_df.to_csv(os.path.join(output_root_dir, "all_clip_results_large_v3", "merged_all_results.csv"), index=False)
