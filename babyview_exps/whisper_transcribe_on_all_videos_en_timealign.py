import torch
from glob import glob
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
import stable_whisper
import logging
logging.getLogger().setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Extract MP3 audio from video files using ffmpeg.")
    parser.add_argument("--mp3_folder", type=str, required=True, help="Folder to save extracted MP3 files.")
    parser.add_argument("--english_subjects_file", type=str, required=False, default="", help="File to save extracted transcripts.")
    parser.add_argument("--transcript_output_folder", type=str, required=True, help="Folder to save extracted transcripts.")
    parser.add_argument("--rank_id", type=int, default=0, help="Rank ID for distributed running.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--is_saycam", type=int, default=0, help="Whether the videos are from SayCam.")
    args = parser.parse_args()
    mp3_folder = args.mp3_folder
    is_saycam = args.is_saycam
    transcript_output_folder = args.transcript_output_folder
    device = torch.device("cuda")
    rank_id = args.rank_id
    num_parallel = args.num_parallel
    model = stable_whisper.load_model('large-v3', device=device)

    english_subjects = []
    filter_english_subjects = False
    english_subjects_file = args.english_subjects_file
    if os.path.exists(english_subjects_file):
        filter_english_subjects = True
        with open(english_subjects_file, 'r') as f:
            for line in f:
                english_subjects.append(line.strip())
    else:
        filter_english_subjects = False
       
    all_audio_files = sorted(glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True))
    en_audio_files = []
    file_name_subject_id = {}
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
        file_name_subject_id[file_name] = subject_id
        if subject_id in english_subjects and filter_english_subjects:
            en_audio_files.append(audio_file)
    
    if filter_english_subjects:
        all_audio_files = en_audio_files

    group_size = len(all_audio_files) // num_parallel
    start_idx = rank_id * group_size
    end_idx = start_idx + group_size
    if rank_id == num_parallel - 1:
        end_idx = len(all_audio_files)
    current_group_audio_files = all_audio_files[start_idx:end_idx]

    for idx, audio_file in enumerate(tqdm(current_group_audio_files)):
        file_name = os.path.basename(audio_file)
        file_name = re.sub(r"\.mp3$", "", file_name)
        subject_id = file_name_subject_id[file_name]
        result = model.transcribe(audio_file, language='en', suppress_silence=True)
        result_dict = result.to_dict()
        data = []
        for chunk in result_dict['segments']:
            start_time = chunk['start']
            end_time = chunk['end']
            data.append({
                'start_time': start_time,
                'end_time': end_time,
                'text': chunk['text']
            })
        # Save result to csv
        res_df = pd.DataFrame(data)
        output_path = os.path.join(transcript_output_folder, subject_id, f"{file_name}.csv")
        if is_saycam:
            output_path = os.path.join(transcript_output_folder, f"{file_name}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index_label="utterance_no")

if __name__ == "__main__":
    main()

