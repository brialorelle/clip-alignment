import os
from glob import glob
import re
from tqdm import tqdm
import argparse
import torch
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--babyview_folder", type=str, required=True)
args_parser.add_argument("--babyview_transcript_folder", type=str, required=True)
args_parser.add_argument("--output_root_dir", type=str, required=True)
args = args_parser.parse_args()



babyview_video_folder = args.babyview_folder
babyview_transcript_fodler = args.babyview_transcript_folder
output_root_dir = args.output_root_dir

cwd_path = os.getcwd()

os.makedirs(output_root_dir, exist_ok=True)

all_subject_number_list = sorted(os.listdir(babyview_video_folder))
total_number_of_videos = len(glob(os.path.join(babyview_video_folder, "*", "*.MP4")))
print(f"The total number of videos found in the main folder: {total_number_of_videos}")

tasks = []
for subjcet in all_subject_number_list:
    all_mp4_files = glob(os.path.join(babyview_video_folder, subjcet, "*.MP4"))
    for mp4_full_path in all_mp4_files:
        mp4_filename = os.path.basename(mp4_full_path)
        transcript_filename = re.sub(r"\.MP4$", ".csv", mp4_filename)
        transcript_full_path = os.path.join(babyview_transcript_fodler,subjcet, transcript_filename)
        output_full_path = os.path.join(output_root_dir, subjcet)
        os.makedirs(output_full_path, exist_ok=True)
        if os.path.exists(transcript_full_path):
            tasks.append((mp4_full_path, transcript_full_path, output_full_path))
        else:
            print(f"[WARNING]: Transcript file not found for {mp4_filename}")
    
for task in tqdm(tasks):
    video_file, csv_file, save_root_dir = task
    os.system(f"python3 {cwd_path}/extract_all_frames.py --video_file {video_file} --csv_file {csv_file} --save_root_dir {save_root_dir}")