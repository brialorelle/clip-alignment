import sys
# sys.path.append("/Users/marielleib/Documents/GitHub/clip-alignment/analysis/babyview_whisper_test")
sys.path.append("/home/bria/clip-alignment/analysis/babyview_whisper_test")
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
def process_video(video_path, frames_dir, csv_dir, mod_dir, whisper_dir, whisper=True):
    # extract video name with which to name our output data
    parts = video_path.split('/')
    video_name, extension = os.path.splitext(parts[-1])

    # Create a new file in the same directory
    file_path = os.path.join(csv_dir, video_name)
    output_csv = file_path + ".csv"
    print(f"File created at: {output_csv}")
    print(f"Processing video: {video_path}")
    
    if whisper:
        # obtain whisper timestamps and utterances
        apply_whisper(video_path, video_name, whisper_dir)
        print("applied whisper")

    # get name of output file so the rest of the pipeline knows what to work with
    file_path = os.path.join(whisper_dir, video_name)
    init_json = file_path + ".json"
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
    # cleaning_dirs(whisper_dir)

def scaling(whisper=True):
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
    # whisper_dir =  os.path.join("/data/babyview/transcripts/Babyview_Main/00230001")
    
    # identify the videos we're working with
    video_dir =  os.path.join("/data/babyview/Babyview_Main/00230001")
    video_extensions = ['MP4']  # Add more extensions if needed
    # apply the pipeline to each video in the videos directory
    for ext in video_extensions:
        pattern = os.path.join(video_dir, f'*.{ext}')
        video_files = glob.glob(pattern)
        
        for video_file in video_files:
            process_video(video_file, frames_dir, csv_dir, mod_dir, whisper_dir, whisper)

if __name__ == "__main__":
    # Functionality to turn off whisper. Pass in false or call the code in terminal "python scale_up.py false" and it won't run whisper
    # Otherwise the default is true do run whisper
    # You MUST already have a whisper output json file in code_files>whisper_output to run without whisper
    # Otherwise it will crash
    
    # Check if there is a command line argument
    if len(sys.argv) > 1:
        # Check if the argument is "False"
        if sys.argv[1].lower() == "false":
            whisper = False
        else:
            whisper = True
    else:
        whisper = True

    scaling(whisper)