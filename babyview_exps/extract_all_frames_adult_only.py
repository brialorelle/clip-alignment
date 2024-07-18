import csv
import os
import subprocess
import argparse
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--video_file", type=str, required=True)
args_parser.add_argument("--csv_file", type=str, required=True)
args_parser.add_argument("--save_root_dir", type=str, required=False)
args = args_parser.parse_args()
# Main script starts here
csv_file = args.csv_file
video_file = args.video_file
# name without extension
video_file_name = os.path.splitext(os.path.basename(video_file))[0]
save_root_dir = args.save_root_dir
if save_root_dir is None:
    save_root_dir = "./"

# Function to extract frames from a video file
def extract_frames(video_file, start_time, end_time, utterance_no):
    # Create a directory for the utterance_no if it doesn't exist
    output_dir = os.path.join(save_root_dir, f'output_frames/{video_file_name}/{utterance_no}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Format ffmpeg command
    # This command extracts one frame per second as an example. You can adjust the 'fps' value as needed.
    command = [
        'ffmpeg',
        '-ss', str(start_time),  # Start time
        '-to', str(end_time),  # End time
        '-i', video_file,  # Input file
        '-vf', 'fps=1',  # Extract one frame per second
        f'{output_dir}/frame_%04d.jpg'  # Output file pattern
    ]
    
    # Execute ffmpeg command
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# Read the CSV file
with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        if row['speaker'] in ['FEM', 'MAL']:
            extract_frames(video_file, row['start_time'], row['end_time'], row['utterance_no'])
        else:
            print(f"[{os.path.basename(csv_file)}]:\nSkipping non-adult utterance<utterance_no,speaker>: <{row['utterance_no'],row['speaker']}>")
            continue
