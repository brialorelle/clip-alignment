from clip_client import Client
c = Client('grpc://0.0.0.0:51000')
import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm.rich import tqdm

# get images for this video/participant
# image_directory = '/Users/brialong/Documents/GitHub/clip-alignment/data/xs-face/frames/'
image_directory = '/data/yinzi/xs_face/frames_full_working'
# List of image extensions
image_extensions = {".jpg"}
# List to store the paths of images
images = glob(os.path.join(image_directory, "**", "*.jpg"), recursive=True)

# import text for this video/participant
all_namings = pd.read_csv('/home/yinzi/workspace/clip-alignment/data/xs-face/all_namings.csv')
utterances = all_namings['utterance']
cor=[]
image_path=[]

for i, this_utterance in tqdm(enumerate(utterances), desc='Calculating clip correlation for each utterance', total=len(utterances)):
	subject_folder = all_namings['child_id'][i]
	object_folder = all_namings['name'][i]
	time = all_namings['time'][i]
	pos_frames_path = os.path.join(image_directory,subject_folder,f'{time}',"pos")
	all_pos_frames = glob(os.path.join(pos_frames_path, "*.jpg"))
	# if positive frames exist
	condition1 = len(all_pos_frames) > 0
	# I found that sometimes the frames are just in the root directory, so I added this condition
	condition2 = len(glob(os.path.join(image_directory,subject_folder,f'{time}',"*.jpg"))) > 0
	if not condition1 and condition2:
		all_pos_frames = glob(os.path.join(image_directory,subject_folder,f'{time}',"*.jpg"))
	# if positive frames exist, calculate the clip correlation, and save the maximum one
	if condition1 or condition2:
		this_text_embedding = c.encode([this_utterance])
		this_text_embedding /= np.linalg.norm(this_text_embedding) # Normalization
		temp_cor = []
		for pos_frame in all_pos_frames:
			this_image_embedding = c.encode([pos_frame])
			this_image_embedding /= np.linalg.norm(this_image_embedding)
			clip_correlation = np.dot(this_text_embedding, this_image_embedding.T)
			temp_cor.append(clip_correlation)
		max_cor = np.max(temp_cor).item()
		cor.append(max_cor)
		image_path.append(pos_frame)
	else:
		cor.append('NA')
		image_path.append('NA')
all_namings['clip_cor']=cor
all_namings['image_path']=image_path
df = pd.DataFrame(all_namings)
df.to_csv('../../all_namings_with_cor.csv', index=False)
# # desired output - csv with columsn that have subject ID, image name, utterance text, r-value, & any other metadata
