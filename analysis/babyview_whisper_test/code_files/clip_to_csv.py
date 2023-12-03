import sys
print(sys.executable)
import pandas as pd
import numpy as np
import os 
from clip_client import Client

# run clip and input result into csv
def clip_service(csv_path, image_dir):
    # set up clip requirements
    c = Client('grpc://0.0.0.0:51000')
    items = pd.read_csv(csv_path)
    print("Now applying clip")

    # set up collumns in csv
    utterance = items['utterance']
    image_path = items['image_path']
    video_name = []
    timestamp = []
    utterance_num = []
    r_value = []

    # These are the data we want in our output csv
    # video_name, utterance_num, timestamp, image_name (utterance_number_timestamp.jpg), utterance_text, r_value
    for i, this_utterance in enumerate(utterance):
        print ("Utterance processed by clip: ", i)
        image_instance = image_path[i]

        # skip if we have already run this before and there is no new information to add
        if 'r_value' in items.columns:
            if items.notnull().all().all():
                break

        # Get the name of the video, utterance, and timestamp from the frame and prep to put in csv
        parts = image_instance.split('_')
        vid_name = '_'.join(parts[0:-2])
        video_name.append(vid_name)
        utterance_num.append(parts[-1].split('.')[0])
        timestamp.append(parts[-2])

        # encode the image and utterance to get the embeddings from which to get the r-values
        this_image = os.path.join(image_dir, image_instance)
        utterance_embeddings = c.encode([this_utterance])
        im_embeddings = c.encode([this_image])
        r_value.append(np.corrcoef(utterance_embeddings, im_embeddings)[0,1])
        print("r val: ", r_value)

    # put all relevant info in the csv
    if video_name:
        items['video_name'] = video_name
    if utterance_num:
        items['utterance_num'] = utterance_num
    if timestamp:
        items['timestamp'] = timestamp
    if r_value:
        items['r_value'] = r_value

    items.to_csv(csv_path, index=False)

    print("CSV is updated")


if __name__ == "__main__":
    clip_service()