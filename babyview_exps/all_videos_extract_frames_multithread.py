import os
from glob import glob
import re
from tqdm import tqdm
import argparse
import torch
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os


def run_extraction(task):
    """
    Executes the extract_all_frames.py script for a single video file.
    Each 'task' is a tuple containing (video_file, csv_file, save_root_dir).
    """
    video_file, csv_file, save_root_dir = task
    cmd = [
        'python', 'extract_all_frames.py',
        '--video_file', video_file,
        '--csv_file', csv_file,
        '--save_root_dir', save_root_dir
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--babyview_folder", type=str, required=True)
    args_parser.add_argument("--babyview_transcript_folder", type=str, required=True)
    args_parser.add_argument("--output_root_dir", type=str, required=True)
    args_parser.add_argument("--max_workers", type=int, default=10)
    args = args_parser.parse_args()
    babyview_video_folder = args.babyview_folder
    babyview_transcript_fodler = args.babyview_transcript_folder
    output_root_dir = args.output_root_dir
    max_workers = args.max_workers

    cwd_path = os.getcwd()
    os.makedirs(output_root_dir, exist_ok=True)
    all_subject_number_list = sorted(os.listdir(babyview_video_folder))
    total_number_of_videos = len(glob(os.path.join(babyview_video_folder, "*", "*.MP4")))
    print(f"The total number of videos found in the main folder: {total_number_of_videos}")
    tasks = []
    for subjcet in tqdm(all_subject_number_list):
        all_mp4_files = sorted(glob(os.path.join(babyview_video_folder, subjcet, "*.MP4")))
        for mp4_full_path in tqdm(all_mp4_files):
            mp4_filename = os.path.basename(mp4_full_path)
            transcript_filename = re.sub(r"\.MP4$", ".csv", mp4_filename)
            transcript_full_path = os.path.join(babyview_transcript_fodler,subjcet, transcript_filename)
            output_full_path = os.path.join(output_root_dir, subjcet)
            os.makedirs(output_full_path, exist_ok=True)
            if os.path.exists(transcript_full_path):
                tasks.append((mp4_full_path, transcript_full_path, output_full_path))
            else:
                print(f"[WARNING]: Transcript file not found for {mp4_filename}")

    progress_bar = tqdm(total=len(tasks))
    # Using ThreadPoolExecutor to run multiple instances of the script in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor: # Adjust max_workers based on your system
        # list(tqdm(executor.map(run_extraction, tasks), total=len(tasks), desc="Extracting Videos"))
        futures = [executor.submit(run_extraction, task) for task in tasks]
        for future in as_completed(futures):
            progress_bar.update(1)


if __name__ == "__main__":
    main()