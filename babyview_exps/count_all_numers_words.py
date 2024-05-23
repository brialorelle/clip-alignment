# %% merge all transcripts into one file
#%%
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

# Paths setup
is_saycam = False
if is_saycam:
    csv_root_dir = "/data/yinzi/saycam/transcripts_distil-large-v3"
    output_root_dir = "/data/yinzi/saycam"
    all_csv_files = glob(os.path.join(csv_root_dir, "*.csv"))
    print(f"Total number of csv files: {len(all_csv_files)}")
    # merge all the csv files into one
    all_saycam_df = pd.DataFrame()
    for csv_file in tqdm(all_csv_files, desc="Processing csv files"):
        single_csv_df = pd.read_csv(csv_file)
        all_saycam_df = pd.concat([all_saycam_df, single_csv_df])
    all_saycam_df.to_csv(os.path.join(output_root_dir, "merged_all_whisper_transcript_largev3.csv"), index=False)
else:
    csv_root_dir = "/data/yinzi/babyview_20240507/transcripts_distil-large-v3/Babyview_Main"
    output_root_dir = "/data/yinzi/babyview_20240507"
    all_subject_number_list = sorted(os.listdir(csv_root_dir))
    excluded_subject =["Bria_Long", "Erica_Yoon"]
    # exclude 
    exclued_all_subject_number_list = [subject for subject in all_subject_number_list if subject not in excluded_subject]
    all_subject_df = pd.DataFrame()
    subject_counts = {}
    for subject in tqdm(exclued_all_subject_number_list, desc="Processing Subjects"):
        subject_csv_files = glob(os.path.join(csv_root_dir, subject, "*.csv"))
        subject_df = pd.DataFrame()
        subject_counts[subject] = len(subject_csv_files)
        for csv_file in tqdm(subject_csv_files, desc="Processing csv files"):
            single_csv_df = pd.read_csv(csv_file)
            subject_df = pd.concat([subject_df, single_csv_df])
        all_subject_df = pd.concat([all_subject_df, subject_df])
    all_subject_df.to_csv(os.path.join(output_root_dir, "merged_all_whisper_transcript_largev3.csv"), index=False)
    sum_all_subject_csv_counts = sum(subject_counts.values())
    print(f"Total number of csv files: {sum_all_subject_csv_counts}")
# %%
# Load your DataFrame
merged_df = pd.read_csv(os.path.join(output_root_dir, "merged_all_whisper_transcript_largev3.csv"))
# %%
import spacy
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm.auto import tqdm

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Extract words (excluding punctuation)
    words = [token.text.lower() for token in doc if token.is_alpha]
    return words

def process_texts(texts, num_workers):
    # Function to process a list of texts and return their tokens
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Processing texts in parallel
        result = list(tqdm(executor.map(tokenize, texts), total=len(texts)))
    return result
# %%
# Specify the number of workers
num_workers = 32  # You can adjust this based on your machine's CPU cores

all_utterance = merged_df['text'].tolist()
all_unqiue_utterance = list(set(all_utterance))
# Flatten the list of lists into a single list of words using multi-processing
spilt_words_of_utterance = process_texts(all_unqiue_utterance, num_workers)
flatten_words = [word for words in spilt_words_of_utterance for word in words]
print(f"Total number of words: {len(flatten_words)}")
# %%
from collections import Counter
# Count the words and get the most common 100
word_freq = Counter(flatten_words)
top_100_words = word_freq.most_common(100)

# Prepare data for plotting
words, counts = zip(*top_100_words)

# Create a histogram
plt.figure(figsize=(15, 10))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
if is_saycam:
    plt.title('Top 100 Word Frequencies SayCam')
else:
    plt.title('Top 100 Word Frequencies BabyView')
plt.xticks(rotation=90)  # Rotate labels to make them readable
plt.show()
