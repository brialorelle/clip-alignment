
# bll bria@stanford.edu 11/2023
# followed instructions using https://github.com/jina-ai/clip-as-service/tree/main/server

# before starting, run
# python3 -m clip_server
# python3

# set up libraries and clip servier
#from clip_client import Client
#c = Client('grpc://0.0.0.0:51000')
#import pandas as pd
#import numpy as np
#import os

# get images for this video/participant
#image_directory = '/Users/brialong/Documents/GitHub/clip-alignment/data/xs-face/transcript_frames/XS_0801/'
# List of image extensions
#image_extensions = {".jpg"}
# List to store the paths of images
#images = []
# Iterate over all files in the directory
#for filename in os.listdir(image_directory):
#    if any(filename.lower().endswith(ext) for ext in image_extensions):
#        images.append(os.path.join(image_directory, filename))


# import text for this video/participant
#text = pd.read_csv('/Users/brialong/Documents/GitHub/clip-alignment/data/xs-face/naming_with_captions/XS_0801_naming_with_captions.csv')
#utterances = text['father_speech']

#for i, this_utterance in enumerate(utterances):
#    print (i)
#    this_image = images[i]
#    this_image_embedding = c.encode([this_image])
#    this_text_embedding = c.encode([this_utterance])
#    clip_correlation = np.corrcoef(this_text_embedding, this_image_embedding)[0,1]
#    print(clip_correlation)


# desired output - csv with columsn that have subject ID, image name, utterance text, r-value, & any other metadata



import sys
print(sys.executable)

import pandas as pd
import numpy as np
import os 
from clip_client import Client


def clip_service(csv_path, image_dir):

    c = Client('grpc://0.0.0.0:51000')
    items = pd.read_csv(csv_path)
    print("Now applying clip")
    utterance = items['utterance']
    image_path = items['image_path']

    video_name = []
    timestamp = []
    utterance_num = []
    r_value = []


    # video_name, utterance_num, timestamp, image_name (utterance_number_timestamp.jpg), utterance_text, r_value

    for i, this_utterance in enumerate(utterance):
        print ("Utterance processed by clip: ", i)
        image_instance = image_path[i]

        # Split the string by underscores
        parts = image_instance.split('_')
        vid_name = '_'.join(parts[0:-2])
        video_name.append(vid_name)
        utterance_num.append(parts[-1].split('.')[0])
        timestamp.append(parts[-2])

        this_image = os.path.join(image_dir, image_instance)
        utterance_embeddings = c.encode([this_utterance])
        im_embeddings = c.encode([this_image])
        r_value.append(np.corrcoef(utterance_embeddings, im_embeddings)[0,1])
        print("r val: ", r_value)


    items['video_name'] = video_name
    items['utterance_num'] = utterance_num
    items['timestamp'] = timestamp
    items['r_value'] = r_value

    items.to_csv(csv_path, index=False)

    print("CSV is updated")


if __name__ == "__main__":
    clip_service()