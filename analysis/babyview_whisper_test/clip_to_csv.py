
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
from memory_profiler import profile

@profile
def clip_service():

    c = Client('grpc://0.0.0.0:51000')

    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_directory, 'output_frames_decimate')

    csv_path = os.path.join(script_directory, 'output.csv')
    items = pd.read_csv(csv_path)

    utterance = items['utterance']
    im_path = items['image_path']

    for i, this_utterance in enumerate(utterance):
        print (i)
        image_instance = im_path[i]
        this_image = os.path.join(image_dir, image_instance)
        print(this_image)
        utterance_embeddings = c.encode([this_utterance])
        print(utterance_embeddings)
        im_embeddings = c.encode([this_image])
        print(im_embeddings)
        clip_correlation = np.corrcoef(utterance_embeddings, im_embeddings)[0,1]
        print(clip_correlation)

    #similarity_score = c.score(utterance, im_path)

    #print(similarity_score)

if __name__ == "__main__":
    clip_service()