import os
from glob import glob
import re
from tqdm import tqdm
import argparse

# Setup argument parser
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--babyview_folder", type=str, required=True)
args_parser.add_argument("--babyview_transcript_folder", type=str, required=True)
args_parser.add_argument("--output_root_dir", type=str, required=True)
args_parser.add_argument("--frame_batch_size", type=int, default=512)
args_parser.add_argument("--clip_batch_size", type=int, default=256)
args_parser.add_argument("--prefetch", type=int, default=100)
args = args_parser.parse_args()

babyview_video_folder = args.babyview_folder
babyview_transcript_folder = args.babyview_transcript_folder
output_root_dir = args.output_root_dir
frame_batch_size = args.frame_batch_size
clip_batch_size = args.clip_batch_size
prefetch = args.prefetch

cwd_path = os.getcwd()
os.makedirs(output_root_dir, exist_ok=True)
all_subject_number_list = sorted(os.listdir(babyview_video_folder))
total_number_of_videos = len(glob(os.path.join(babyview_video_folder, "*", "*.MP4")))
print(f"The total number of videos found in the main folder: {total_number_of_videos}")

def process_video(mp4_full_path, subject):
    mp4_filename = os.path.basename(mp4_full_path)
    if re.search(r"\.MP4$", mp4_filename):
        partten = r"\.MP4$"
    else:
        partten = r"\.mp4$"
    transcript_filename = re.sub(partten, ".csv", mp4_filename)
    transcript_full_path = os.path.join(babyview_transcript_folder, subject, transcript_filename)
    output_full_path = os.path.join(output_root_dir, subject)
    os.makedirs(output_full_path, exist_ok=True)

    if not os.path.exists(transcript_full_path):
        print(f"[WARNING]: Transcript file not found for {mp4_filename}")
        return
    
    # Replace with the correct script path and its arguments
    # command = f"python3 {cwd_path}/clip_on_all_frames.py --video_file {mp4_full_path} --csv_file {transcript_full_path} --save_root_dir {output_full_path}"
    command = f"python3 {cwd_path}/batch_clip_on_all_frames.py --video_file {mp4_full_path} --csv_file {transcript_full_path} --save_root_dir {output_full_path} --frame_batch_size {frame_batch_size} --clip_batch_size {clip_batch_size} --prefetch {prefetch}"
    os.system(command)

# Process each video in a single-threaded manner
progress_bar = tqdm(total=total_number_of_videos)
for subject in all_subject_number_list:
    print("Current Subject:", subject)
    all_mp4_files = glob(os.path.join(babyview_video_folder, subject, "*.MP4"))
    for mp4_full_path in all_mp4_files:
        process_video(mp4_full_path, subject)
        progress_bar.update(1)
