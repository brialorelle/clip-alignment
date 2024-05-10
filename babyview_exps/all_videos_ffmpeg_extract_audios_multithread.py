import os
from glob import glob
import subprocess
import argparse
import concurrent.futures
from tqdm import tqdm

def extract_audio(video_path, mp3_path):
    """
    Uses ffmpeg to extract MP3 audio from a video file.
    """
    cmd = [
        'ffmpeg',
        '-i', video_path,    # Input video path
        '-y',                # Overwrite
        '-vn',               # Disable video processing
        '-ar', '44100',      # Set audio sampling rate to 44100 Hz
        '-ac', '2',          # Set number of audio channels to 2
        '-b:a', '192k',      # Set audio bitrate to 192 kbps
        mp3_path             # Output MP3 path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    parser = argparse.ArgumentParser(description="Extract MP3 audio from video files using ffmpeg.")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing video files.")
    parser.add_argument("--mp3_folder", type=str, required=True, help="Folder to save extracted MP3 files.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads.")
    args = parser.parse_args()

    video_folder = args.video_folder
    mp3_folder = args.mp3_folder
    max_workers = args.max_workers

    os.makedirs(mp3_folder, exist_ok=True)
    video_files = glob(os.path.join(video_folder, "**", "*.MP4"), recursive=True)
    video_files += glob(os.path.join(video_folder, "**", "*.mp4"), recursive=True)
    print(video_files)

    tasks = []
    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        mp3_filename = os.path.splitext(video_filename)[0] + '.mp3'
        mp3_path = os.path.join(mp3_folder, mp3_filename)
        tasks.append((video_path, mp3_path))

    progress_bar = tqdm(total=len(tasks), desc="Extracting MP3s")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_audio, task[0], task[1]) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            progress_bar.update(1)

if __name__ == "__main__":
    main()