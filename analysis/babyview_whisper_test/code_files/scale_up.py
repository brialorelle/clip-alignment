import sys
sys.path.append("/Users/marielleib/Documents/GitHub/clip-alignment/analysis/babyview_whisper_test")
import os
import glob
from code_files.Transform_JSON import transform
from code_files.apply_whisper_to_videos import apply_whisper
from code_files.clip_to_csv import clip_service
from code_files.extract_frames import extract_frames

def cleaning_dirs(directory_path):
    print(f"Deleting intermediate files {directory_path}")
    # List all files in the directory
    files = os.listdir(directory_path)

    # Iterate through the files and delete them
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file}: {e}")


# Put each video through the pipeline
def process_video(video_path, frames_dir, csv_dir, mod_dir, whisper_dir):
    # extract video name with which to name our output data
    parts = video_path.split('/')
    video_name, extension = os.path.splitext(parts[-1])
    # video_name = parts[-1].split('.')[0]

    # Create a new file in the same directory
    file_path = os.path.join(csv_dir, video_name)
    output_csv = file_path + ".csv"
    print(f"File created at: {output_csv}")
    print(f"Processing video: {video_path}")

    # obtain whisper timestamps and utterances
    init_json = apply_whisper(video_path, video_name, whisper_dir)
    print("apply whisper output: ", init_json)
    
    # extract just utterances and timestamps from output jsons
    json_name = transform(init_json, mod_dir)
    print("name of transformed json file: ", json_name)
     
    # running ffmpeg to get the frames where the utterance is spoken
    extract_frames(video_path, frames_dir, json_name, output_csv)
    
    # run frames through clip to get r-vals (how much the video shows what the utterance mentions) and put in csv
    clip_service(output_csv, frames_dir)

    print("Finished with this video")

    cleaning_dirs(frames_dir)
    cleaning_dirs(mod_dir)
    cleaning_dirs(whisper_dir)

def scaling():
    # get the name of the directory we're working in
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Find the last index of '/'
    last_slash_index = script_directory.rfind('/')
    # Remove everything after the last '/' so we have the parent dir
    dir = script_directory[:last_slash_index + 1]
    print(dir)

    # get the directories for the folders of the output files
    frames_dir = os.path.join(dir, 'output_frames')
    csv_dir = os.path.join(dir, 'csvs')
    mod_dir = os.path.join(dir, 'modified_jsons')
    whisper_dir = os.path.join(dir, 'whisper_output')

    # identify the videos we're working with
    video_dir = os.path.join(dir, 'videos/')
    video_extensions = ['mp4', 'avi', 'mov']  # Add more extensions if needed
    # apply the pipeline to each video in the videos directory
    for ext in video_extensions:
        pattern = os.path.join(video_dir, f'*.{ext}')
        video_files = glob.glob(pattern)
        
        for video_file in video_files:
            process_video(video_file, frames_dir, csv_dir, mod_dir, whisper_dir)



if __name__ == "__main__":
    scaling()