# scale_up.py

import sys
sys.path.append("/Users/marielleib/Documents/GitHub/clip-alignment/analysis/babyview_whisper_test")

import os
import glob
from code_files.Transform_JSON import transform
from code_files.apply_whisper_to_videos import apply_whisper
from code_files.clip_to_csv import clip_service
from code_files.extract_frames import extract_frames



def process_video(video_path, output_dir, output_csv, mod_dir):
    # Your video processing logic goes here
    print(f"Processing video: {video_path}")

    # obtain whisper timestamps and utterances
    init_json = apply_whisper(video_path)
    print("apply whisper output: ", init_json)
    
    # extract just utterances and timestamps from output json
    json_name = transform(init_json, mod_dir)
    print("name of transformed json file: ", json_name)

    # running ffmpeg
    extract_frames(video_path, output_dir, json_name, output_csv)

    # run frames through clip to get r-vals and put in same csv
    #clip_service(output_csv, output_dir)

    print("Finished with this video")



def scaling():
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Find the last index of '/'
    last_slash_index = script_directory.rfind('/')

    # Remove everything after the last '/'
    dir = script_directory[:last_slash_index + 1]
    print(dir)

    video_dir = os.path.join(dir, 'videos/')
    video_extensions = ['mp4', 'avi', 'mov']  # Add more extensions if needed

    output_dir = os.path.join(dir, 'output_frames')
    csv_dir = os.path.join(dir, 'csvs')
    mod_dir = os.path.join(dir, 'modified_jsons')
    # Create a new file in the same directory
    output_csv = os.path.join(csv_dir, 'output.csv')
    print(f"File created at: {output_csv}")

    for ext in video_extensions:
        pattern = os.path.join(video_dir, f'*.{ext}')
        video_files = glob.glob(pattern)
        
        for video_file in video_files:
            process_video(video_file, output_dir, output_csv, mod_dir)



if __name__ == "__main__":
    scaling()