# %%
import os
import re
import csv
import pandas as pd
from tqdm import tqdm
from glob import glob

babyview_video_folder = "/data/yinzi/babyview/Babyview_Main"
output_root_dir = "/data/yinzi/babyview/"
all_subject_number_list = sorted(os.listdir(babyview_video_folder))
# exclude pilot subjects
exlucde_subjects = ['Bria_Long', 'Erica_Yoon']
all_subject_number_list = [subject for subject in all_subject_number_list if subject not in exlucde_subjects]
total_number_of_videos = sum([len(glob(os.path.join(babyview_video_folder, subject, "*.MP4")) ) for subject in all_subject_number_list])


progress_bar = tqdm(total=total_number_of_videos)
all_chosen_frames_df = pd.DataFrame()
for subject in all_subject_number_list:
    # create a single folder
    os.makedirs(os.path.join(output_root_dir, "all_clip_results","chosen_frames"), exist_ok=True)
    all_mp4_files = glob(os.path.join(babyview_video_folder, subject, "*.MP4"))
    subject_df = pd.DataFrame()
    # merge all single mp4 csv files into one, rename the utterance_no in format: {video_name}_{utterance_no}
    for mp4_full_path in all_mp4_files:
        video_file_name = os.path.splitext(os.path.basename(mp4_full_path))[0]
        output_csv_dir = os.path.join(output_root_dir, "all_clip_results",subject, "all_result_csv_files", video_file_name,'clip_final_results.csv')
        single_video_df = pd.read_csv(output_csv_dir)
        # skip Nan values
        single_video_df = single_video_df.dropna(subset=['dot_product'])
        # rename the utterance_no in format: {video_name}_{utterance_no}
        single_video_df['utterance_no'] = single_video_df['utterance_no'].apply(lambda x: f"{video_file_name}_{x}")
        # merge
        subject_df = pd.concat([subject_df, single_video_df], ignore_index=True)
    utterances = single_video_df['utterance_no'].unique()
    results = []
    for utterance_no in utterances:
        utterance_df = single_video_df[single_video_df['utterance_no'] == utterance_no]
        max_frame = utterance_df.loc[utterance_df['dot_product'].idxmax()]
        min_frame = utterance_df.loc[utterance_df['dot_product'].idxmin()]
        max_dot_product = max_frame['dot_product']
        min_dot_product = min_frame['dot_product']
        if max_dot_product == min_dot_product:
            # print(f"Same dot product for {utterance_no}")
            continue
        result = {
            "utterance_no": utterance_no,
            "max_frame": max_frame['frame'],
            "max_dot_product": max_dot_product,
            "min_frame": min_frame['frame'],
            "min_dot_product": min_dot_product
        }
        results.append(result)
    
    extreme_frame_df = pd.DataFrame(results)
    sample_number = 10
    if len(extreme_frame_df) < sample_number:
        print(f"Less than 10 valid utterances for {subject}, only {len(extreme_frame_df)} valid utterances.")
        sample_number = len(extreme_frame_df)
        print(len(utterances))
        print(len(all_mp4_files))
        print(output_csv_dir)
    # random sample some utterances for each subject
    chosen_frame_df = extreme_frame_df.sample(sample_number)
    all_chosen_frames_df = pd.concat([all_chosen_frames_df, chosen_frame_df], ignore_index=True)
#%%
    
#%% 
from shutil import copy2

save_frame_dir = os.path.join(output_root_dir, "all_clip_results","chosen_frames")
os.makedirs(save_frame_dir, exist_ok=True)
# save the chosen highest and lowest dot product frames for each subject
all_paths = []
for index, row in all_chosen_frames_df.iterrows():
    utterance_no = row['utterance_no']
    max_frame_source_path = row['max_frame']
    max_frame_score = row['max_dot_product']
    min_frame_source_path = row['min_frame']
    min_frame_score = row['min_dot_product']

    max_frame_dest_path = os.path.join(save_frame_dir, f"{utterance_no}_{index}_clip_{max_frame_score}.jpg")
    min_frame_dest_path = os.path.join(save_frame_dir, f"{utterance_no}_{index}_clip_{min_frame_score}.jpg")
    all_paths.append(max_frame_source_path)
    all_paths.append(min_frame_source_path)
    # copy the frame to the save_frame_dir
    copy2(max_frame_source_path, max_frame_dest_path)
    copy2(min_frame_source_path, min_frame_dest_path)

    
# %%
