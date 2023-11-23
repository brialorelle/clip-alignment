import json
import subprocess
from datetime import datetime
import os
import csv

def timecode_to_seconds(timecode):
    # Convert timecode in HH:MM:SS format to seconds
    time_object = datetime.strptime(timecode, '%H:%M:%S')
    return time_object.second + time_object.minute * 60 + time_object.hour * 3600

def extract_frames(input_video, output_folder, json_file, output_csv):
    try:
        with open(json_file) as f:
            data = json.load(f)
        print("Now extracting frames with ffmpeg")
        # Your code to create or manipulate the file can go here
        with open(output_csv, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # If the file is empty, write the header row
            if csv_file.tell() == 0:
                csv_writer.writerow(['utterance', 'image_path'])
                
            for key, value in data.items():

                start_time = value['start']
                end_time = value['end']
                
                # set up where the frames will be put
                parts = input_video.split('/')
                filename = parts[-1].split('.')[0]
                name = filename + '_' + value["start"] + '_%02d.png'
                output_file = os.path.join(output_folder, name)

                # Run FFmpeg command to extract frames with explicit pixel format
                cmd = [
                    'ffmpeg', 
                    #'-v', "verbose",
                    
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-i', input_video,
                    '-vf', 'fps=1, format=rgb24',
                    '-vsync', 'vfr',
                    output_file
                ]
                print("Ffmpeg command: ", cmd)
                subprocess.run(cmd, check=True)  # Add check=True to raise an error if FFmpeg command fails
                
                # List all files in the directory
                files = os.listdir(output_folder)

                # Print each filename
                print(f"Filenames in '{output_folder}':")
                for filename in files:
                    if filename.endswith(".png") and start_time in filename:
                        print(filename)
                        print(value['utterance'])
                        csv_writer.writerow([value['utterance'], filename])

        print(f'CSV file "{output_csv}" has been edited.')

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # You can choose to log the error or perform any other necessary cleanup/termination here
        raise  # Re-raise the exception to propagate it up the call stack


if __name__ == "__main__":
    # Get the absolute path to the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Get the user's home directory
    home_directory = os.path.expanduser("~")

    # Path to the Downloads folder
    downloads_folder = os.path.join(home_directory, script_directory)
    input_video = os.path.join(downloads_folder, 'output.mp4')
    output_folder = os.path.join(downloads_folder, 'output_frames_decimate')
    json_file = os.path.join(downloads_folder, 'output_mod.json')

    path = os.path.join(downloads_folder, "output.json")


    #input_video = '/Users/marielleib/Documents/GitHub/clip-alignment/example_clip.mov'  # Replace with your input video file path
    #output_folder = '/Users/marielleib/Documents/GitHub/clip-alignment/output_frames'  # Replace with your desired output folder path
    #json_file = '/Users/marielleib/Documents/GitHub/clip-alignment/example_clip_mod.json'  # Replace with your JSON file path

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        extract_frames(input_video, output_folder, json_file, script_directory)
    except Exception as e:
        print(f"Failed to extract frames: {str(e)}")
        # You can choose to log the error or perform any other necessary cleanup/termination here
