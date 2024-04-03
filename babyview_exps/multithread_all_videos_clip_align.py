import os
from glob import glob
import re
from tqdm import tqdm
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

# Setup argument parser
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--babyview_folder", type=str, required=True)
args_parser.add_argument("--babyview_transcript_folder", type=str, required=True)
args_parser.add_argument("--output_root_dir", type=str, required=True)
args_parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads")
args = args_parser.parse_args()

babyview_video_folder = args.babyview_folder
babyview_transcript_folder = args.babyview_transcript_folder
output_root_dir = args.output_root_dir
max_workers = args.max_workers

cwd_path = os.getcwd()
os.makedirs(output_root_dir, exist_ok=True)
all_subject_number_list = sorted(os.listdir(babyview_video_folder))
total_number_of_videos = len(glob(os.path.join(babyview_video_folder, "*", "*.MP4")))
print(f"The total number of videos found in the main folder: {total_number_of_videos}")

def process_video(mp4_full_path, subject):
    mp4_filename = os.path.basename(mp4_full_path)
    transcript_filename = re.sub(r"\.MP4$", ".csv", mp4_filename)
    transcript_full_path = os.path.join(babyview_transcript_folder, subject, transcript_filename)
    output_full_path = os.path.join(output_root_dir, subject)
    os.makedirs(output_full_path, exist_ok=True)

    if not os.path.exists(transcript_full_path):
        print(f"[WARNING]: Transcript file not found for {mp4_filename}")
        return
    
    # Replace with the correct script path and its arguments
    command = f"python3 {cwd_path}/clip_on_all_frames.py --video_file {mp4_full_path} --csv_file {transcript_full_path} --save_root_dir {output_full_path}"
    os.system(command)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all video processing tasks to the executor
    futures = []
    for subject in all_subject_number_list:
        all_mp4_files = glob(os.path.join(babyview_video_folder, subject, "*.MP4"))
        for index, mp4_full_path in enumerate(all_mp4_files):
            futures.append(executor.submit(process_video, mp4_full_path, subject))

    progress_bar = tqdm(total=len(futures))
    # Progress bar for futures
    for future in concurrent.futures.as_completed(futures):
        progress_bar.update(1)
        future.result() # This line is optional, it will raise exceptions if any occurred during the function execution
