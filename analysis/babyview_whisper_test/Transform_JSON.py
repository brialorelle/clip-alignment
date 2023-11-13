import json
import os

# Function to convert float time to timestamp format - thanks chatgpt 
def float_to_timestamp(float_time):
    seconds, miliseconds = divmod(float_time * 60, 60)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(minutes)
    print(seconds)
    return "{:02}:{:02}:{:02}.{:02}".format(int(hours), int(minutes), int(seconds), int(miliseconds))


def transform(file):
    with open(file, 'r') as file:
        data = json.load(file)

    parsed = data["segments"]
    timestamps = {}
    key = 0
    for entry in parsed:
        key += 1
        start = entry["start"]
        start = float_to_timestamp(start)
        stop = entry["end"]
        stop = float_to_timestamp(stop)
        utterance = entry["text"]
        
        timestamps[key] = {"start": start, "end": stop, "utterance": utterance}

    print(timestamps)

    with open ('example_clip_mod.json', 'w') as json_file:
        json.dump(timestamps, json_file)



if __name__ == "__main__":
    # Get the user's home directory
    home_directory = os.path.expanduser("~")

    # Path to the Downloads folder
    downloads_folder = os.path.join(home_directory, "Downloads")
    path = os.path.join(downloads_folder, "example_clip.json")

    transform(path)