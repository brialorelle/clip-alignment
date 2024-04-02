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
# babyview_video_folder = "/data/babyview/Babyview_Main"
# babyview_transcript_fodler = "/data/babyview/transcripts/Babyview_Main"
# output_root_dir = "/data/babyview/all_clip_results"

os.makedirs(output_root_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

all_subject_number_list = sorted(os.listdir(babyview_video_folder))

total_number_of_videos = len(glob(os.path.join(babyview_video_folder, "*", "*.MP4")))
print(f"The total number of videos found in the main folder: {total_number_of_videos}")

for subjcet in tqdm(all_subject_number_list):
    all_mp4_files = glob(os.path.join(babyview_video_folder, subjcet, "*.MP4"))
    for mp4_full_path in tqdm(all_mp4_files):
        mp4_filename = os.path.basename(mp4_full_path)
        transcript_filename = re.sub(r"\.MP4$", ".csv", mp4_filename)
        transcript_full_path = os.path.join(babyview_transcript_fodler,subjcet, transcript_filename)
        output_full_path = os.path.join(output_root_dir, subjcet)
        os.makedirs(output_full_path, exist_ok=True)
        os.system(f"python3 {cwd_path}/extract_all_frames.py --video_file {mp4_full_path} --csv_file {transcript_full_path} --save_root_dir {output_full_path}")
        os.system(f"python3 {cwd_path}/clip_on_all_frames.py --video_file {mp4_full_path} --csv_file {transcript_full_path} --device {device} --save_root_dir {output_full_path}")
        # Skip if transcript file for this video not found
        try:
            assert os.path.exists(transcript_full_path)

        except:
            print(f"[WARNING]: Transcript file not found for {mp4_filename}")
            continue
    