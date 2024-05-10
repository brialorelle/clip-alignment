import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from glob import glob
import pandas as pd
import os
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Process audio files using Whisper.")
    parser.add_argument("--mp3_folder", type=str, required=True, help="Folder containing MP3 files.")
    parser.add_argument("--transcript_output_folder", type=str, required=True, help="Folder to save transcripts.")
    parser.add_argument("--device_ids", type=str, default="[0,1,2,3]", help="List of GPU device IDs to use.")
    parser.add_argument("--num_parallel", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--is_saycam", type=int, default=0, help="Whether the videos are from SayCam.")
    args = parser.parse_args()

    mp3_folder = args.mp3_folder
    transcript_output_folder = args.transcript_output_folder
    device_ids = [int(id) for id in args.device_ids.strip("[]").split(",")]
    num_devices = len(device_ids)
    num_parallel = args.num_parallel
    is_saycam = args.is_saycam

    all_audio_files = glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True)

    # Ensure the transcript output folder exists
    os.makedirs(transcript_output_folder, exist_ok=True)

    all_rank_ids = list(range(num_parallel))
    rank_device_dict = {rank_id: device_ids[rank_id % num_devices] for rank_id in all_rank_ids}

    session_name_with_random_suffix = f"whisper_transcribe_{os.urandom(4).hex()}"
    session_name = session_name_with_random_suffix

    # Create a new tmux session and split into the required number of panes
    os.system(f"tmux new-session -d -s {session_name}")
    for i in range(1, num_parallel):
        os.system(f"tmux split-window -t {session_name} -h")
        os.system(f"tmux select-layout -t {session_name} tiled")

    # Send the command to each pane
    for rank_id in all_rank_ids:
        device_id = rank_device_dict[rank_id]
        command = (
            f"conda activate torch_2_2_2;"
            f"python3 whisper_transcribe_on_all_videos.py --mp3_folder {mp3_folder} "
            f"--transcript_output_folder {transcript_output_folder} --device cuda:{device_id} "
            f"--rank_id {rank_id} --num_parallel {num_parallel} --is_saycam {is_saycam}"
        )
        os.system(f"tmux send-keys -t {session_name}.{rank_id} '{command}' Enter")
    print(f"Started {num_parallel} parallel processes in tmux session {session_name}")
    print(f"Use 'tmux attach -t {session_name}' to view progress")

if __name__ == "__main__":
    main()
