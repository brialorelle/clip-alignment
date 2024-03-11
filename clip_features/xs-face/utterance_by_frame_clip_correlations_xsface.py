 
# bll bria@stanford.edu 11/2023
# followed instructions using https://github.com/jina-ai/clip-as-service/tree/main/server

# before starting, run
# python3 -m clip_server
# python3

# set up libraries and clip servier
from clip_client import Client
c = Client('grpc://0.0.0.0:51000')
import pandas as pd
import numpy as np
import os

# get images for this video/participant
image_directory = '/Users/brialong/Documents/GitHub/clip-alignment/data/xs-face/frames/'
# List of image extensions
image_extensions = {".jpg"}
# List to store the paths of images
images = []
# Iterate over all files in the directory
for filename in os.listdir(image_directory):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        images.append(os.path.join(image_directory, filename))


# import text for this video/participant
all_namings = pd.read_csv('/Users/brialong/Documents/GitHub/clip-alignment/data/xs-face/all_namings.csv')
utterances = all_namings['utterance']
cor=[]
image_path=[]

for i, this_utterance in enumerate(utterances):
	subject_folder = all_namings['child_id'][i]
	object_folder = all_namings['name'][i]
	time = all_namings['time'][i]
	this_image = os.path.join(image_directory,subject_folder,object_folder, str(time)+'.jpg')
	short_image_path = os.path.join(subject_folder,object_folder, str(time)+'.jpg')
	if os.path.exists(this_image):
		if np.mod(i,10)==0:
			print(i)
		this_utterance = all_namings['utterance'][i]
		this_image_embedding = c.encode([this_image])
		if pd.isna(this_utterance)==False:
			this_text_embedding = c.encode([this_utterance])
			clip_correlation = np.corrcoef(this_text_embedding, this_image_embedding)[0,1]
			print(clip_correlation)
			cor.append(clip_correlation)
			image_path.append(short_image_path)
		else:
			cor.append('NA') # if there is no annotation
			image_path.append('NA')
	else:
		cor.append('NA') # if there is no image
		image_path.append('NA')

all_namings['clip_cor']=cor
all_namings['short_image_path']=short_image_path
df = pd.DataFrame(all_namings)

df.to_csv('all_namings_with_cor.csv', index=False)


# desired output - csv with columsn that have subject ID, image name, utterance text, r-value, & any other metadata


