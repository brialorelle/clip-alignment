# bll 7.7.22 bria@stanford.edu
# followed instructions using https://github.com/jina-ai/clip-as-service/tree/main/server

# before starting, run
# python3 -m clip_server
# python3

#from clip_client import Client
#c = Client('grpc://0.0.0.0:51000')
#import pandas as pd
#import numpy as np
# has category items

# from ecoset
# items = pd.read_csv('/Users/brialong/Documents/GitHub/online-vocab-assessment/data/ecoset/candidate_items_from_ecoset.csv')
# from things
#items = pd.read_csv('/Users/brialong/Documents/GitHub/online-vocab-assessment/stimuli/selected_stimuli/things_dataset/things_concepts.tsv', sep='\t')
#all_items = list(items['Word'])
#all_items =  all_items

# wow so fast
#item_embeddings = c.encode(all_items)
#embeddings_items = np.size(item_embeddings,0)
#embeddings_vector_length= np.size(item_embeddings,0)


# get correlations and save
#item_correlations = np.corrcoef(item_embeddings)
#item_correlations_df = pd.DataFrame(item_correlations)
#item_correlations_df.columns = all_items
#item_correlations_df_transposed = item_correlations_df.transpose()
#item_correlations_df_transposed.columns = all_items
#item_correlations_df_transposed.to_csv('/Users/brialong/Documents/GitHub/online-vocab-assessment/stimuli/selected_stimuli/things_dataset/things_test_all_item_embeddings.csv')
import sys
print(sys.executable)

import pandas as pd
import numpy as np
import os 
from clip_as_service import Client
from clip_client import Client

c = Client('grpc://0.0.0.0:51000')

script_directory = os.path.dirname(os.path.abspath(__file__))
    
# Get the user's home directory
home_directory = os.path.expanduser("~")

# Path to the Downloads folder
downloads_folder = os.path.join(home_directory, script_directory)
csv_path = os.path.join(downloads_folder, 'output.csv')

items = pd.read_csv(csv_path)
utterance = items['utterance']
im_path = items['image_path']


item_embeddings = c.encode(utterance)
embeddings_items = np.size(item_embeddings,0)
embeddings_vector_length= np.size(item_embeddings,0)

item_embeddings = c.encode(im_path)
embeddings_items = np.size(item_embeddings,0)
embeddings_vector_length= np.size(item_embeddings,0)



similarity_score = c.score(utterance, im_path)

print(similarity_score)