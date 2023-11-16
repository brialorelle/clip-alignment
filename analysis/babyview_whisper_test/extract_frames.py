import json
import subprocess
from datetime import datetime
import os
import csv

def timecode_to_seconds(timecode):
    # Convert timecode in HH:MM:SS format to seconds
    time_object = datetime.strptime(timecode, '%H:%M:%S')
    return time_object.second + time_object.minute * 60 + time_object.hour * 3600

def extract_frames(input_video, output_pattern, json_file):
    try:
        with open(json_file) as f:
            data = json.load(f)
            
        # Specify the CSV file name
        csv_file_name = 'output.csv'

        # Open the CSV file for writing
        with open(csv_file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row to the CSV file
            csv_writer.writerow(['utterance', 'image_path'])


            for key, value in data.items():
                #start_time = timecode_to_seconds(value['start'])
                #end_time = timecode_to_seconds(value['end'])

                start_time = value['start']
                end_time = value['end']

                # Use the 'utterance' field as part of the output filename
                output_file = os.path.join(output_folder, f'output_frames_{value["start"]}_%02d.png')
                
                # Write the entry to the CSV file
                csv_writer.writerow([value['utterance'], output_file])


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
                print(cmd)
                subprocess.run(cmd, check=True)  # Add check=True to raise an error if FFmpeg command fails

                # Write entries to the CSV file for each frame
                for i in range(1, len(data)+1):
                    output_file = output_file.format(i)
                    csv_writer.writerow([value['utterance'], output_file])


        print(f'CSV file "{csv_file_name}" has been created.')

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # You can choose to log the error or perform any other necessary cleanup/termination here
        raise  # Re-raise the exception to propagate it up the call stack


if __name__ == "__main__":
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
        extract_frames(input_video, output_folder, json_file)
    except Exception as e:
        print(f"Failed to extract frames: {str(e)}")
        # You can choose to log the error or perform any other necessary cleanup/termination here
