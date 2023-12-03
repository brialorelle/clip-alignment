import json
import subprocess
from datetime import datetime
import os
import csv

# Convert timecode in HH:MM:SS format to seconds
def timecode_to_seconds(timecode):
    time_object = datetime.strptime(timecode, '%H:%M:%S')
    return time_object.second + time_object.minute * 60 + time_object.hour * 3600

# list the frames that ffmpeg outputted in a csv
def process_files_in_folder(folder_path, target_string, value, csv_writer):
    # go through all the produced frames and use only the ones from this video and timestamp
    for filename in os.listdir(folder_path):
        if target_string in filename:
            file_path = os.path.join(folder_path, filename)

            # list the filename with the associated utterance in the csv
            print(f"Processing file: {file_path}")
            print(value['utterance'])
            csv_writer.writerow([value['utterance'], filename])

# run ffmpeg
def extract_frames(input_video, output_folder, json_file, output_csv):
    # get json data
    with open(json_file) as f:
        data = json.load(f)
    print("Now extracting frames with ffmpeg")
    
    # open the CSV we want to add to
    with open(output_csv, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # If the file is empty, write the header row
        if csv_file.tell() == 0:
            csv_writer.writerow(['utterance', 'image_path'])
            
        # iterate through each utterance timestamp
        for key, value in data.items():
            start_time = value['start']
            end_time = value['end']
            
            # set up where the frames will be put
            parts = input_video.split('/')
            filename = parts[-1].split('.')[0]
            name_wo_numbering = filename + '_' + value["start"]
            name = name_wo_numbering + '_%02d.png'
            output_file = os.path.join(output_folder, name)

            # Run FFmpeg command to extract frames with explicit pixel format
            cmd = [
                'ffmpeg', 
                '-ss', str(start_time),
                '-to', str(end_time),
                '-i', input_video,
                '-vf', 'fps=1, format=rgb24',
                '-vsync', 'vfr',
                output_file
            ]
            print("Ffmpeg command: ", cmd)
            subprocess.run(cmd, check=True)  # Add check=True to raise an error if FFmpeg command fails
    
            # Print each filename
            print(f"Filenames in '{output_folder}':")
            process_files_in_folder(output_folder, name_wo_numbering, value, csv_writer)

    print(f'CSV file "{output_csv}" has been edited.')

if __name__ == "__main__":
    try:
        extract_frames(input_video, output_folder, json_file, script_directory)
    except Exception as e:
        print(f"Failed to extract frames: {str(e)}")
