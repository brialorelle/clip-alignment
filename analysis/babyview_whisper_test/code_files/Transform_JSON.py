import json
import os

# Function to convert float time to timestamp format
def float_to_timestamp(float_time):
    seconds, miliseconds = divmod(float_time * 60, 60)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(minutes)
    print(seconds)
    return "{:02}:{:02}:{:02}.{:02}".format(int(hours), int(minutes), int(seconds), int(miliseconds))

# Takes in a json file outputted by whisper and the output directory and extracts just timestamps and utterances
def transform(file, mod_dir):
    # get name of file
    parts = file.split('/')
    filename, extension = os.path.splitext(parts[-1])
    filename2 = filename + "_mod.json"
    name = os.path.join(mod_dir, filename2)

    # start on json data
    with open(file, 'r') as file:
        data = json.load(file)
    print("Now transforming json")
    parsed = data["segments"]
    timestamps = {}
    key = 0
    # extract desired info for each entry in the json file
    for entry in parsed:
        key += 1
        start = entry["start"]
        start = float_to_timestamp(start)
        stop = entry["end"]
        stop = float_to_timestamp(stop)
        utterance = entry["text"]
        
        timestamps[key] = {"start": start, "end": stop, "utterance": utterance}

    print("Json timestamps extracted: ", timestamps)
    # put narrowed info file into new json file
    with open(name, 'w') as json_file:
        json.dump(timestamps, json_file)
    print("Timestamps put into modified json")
    return name



if __name__ == "__main__":
    transform(file, modified_dir)