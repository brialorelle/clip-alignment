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


def transform(file, mod_dir):
    parts = file.split('/')
    filename = parts[-1].split('.')[0]
    filename2 = filename + "_mod.json"
    name = os.path.join(mod_dir, filename2)

    with open(file, 'r') as file:
        data = json.load(file)
    print("Now transforming json")
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

    print("Json timestamps extracted: ", timestamps)

    with open(name, 'w') as json_file:
        json.dump(timestamps, json_file)
    print("Timestamps put into modified json")
    return name



if __name__ == "__main__":
    # Get the user's home directory
    #home_directory = os.path.expanduser("~")

    # Path to the Downloads folder
    #script_path = os.path.abspath(__file__)
    #parent_folder = os.path.dirname(script_path)
    #downloads_folder = os.path.join(home_directory, parent_folder)
    #path = os.path.join(downloads_folder, "output.json")

    transform(file, mod_dir)