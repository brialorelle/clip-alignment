
# bll bria@stanford.edu 11/2023
# followed instructions using https://github.com/jina-ai/clip-as-service/tree/main/server

# before starting, run
# python3 -m clip_server
# python3


# mostly repeat of clip_to_csv.py, WE ARE NOT USING

import pandas as pd
import numpy as np
import os 
from clip_client import Client

c = Client('grpc://0.0.0.0:51000')


script_directory = os.getcwd()
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
    utterance_embeddings = c.encode([this_utterance])
    im_embeddings = c.encode([this_image])
    clip_correlation = np.corrcoef(utterance_embeddings, im_embeddings)[0,1]
    print(clip_correlation)

