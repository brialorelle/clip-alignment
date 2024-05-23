#%%
import os
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
import filecmp

def csv_to_txt(csv_files, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for file in tqdm(csv_files, desc="Converting CSV to TXT"):
        try:
            df = pd.read_csv(file)
            if 'text' in df.columns:
                texts = df['text'].dropna().tolist()
                txt_file_path = os.path.join(output_folder, os.path.basename(file).replace('.csv', '.txt'))
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    for text in texts:
                        txt_file.write(text + '\n')
        except Exception as e:
            print(f"Error processing file {file}: {e}")

def find_duplicate_txt_files(folder_path, output_file):
    txt_files = glob(folder_path + '/*.txt')
    
    duplicates = []
    compared_pairs = set()
    
    for i in tqdm(range(len(txt_files)), desc="Comparing TXT files"):
        for j in range(i + 1, len(txt_files)):
            file1, file2 = txt_files[i], txt_files[j]
            subject_id_1 = os.path.basename(file1).split('_')[0]
            subject_id_2 = os.path.basename(file2).split('_')[0]
            # only compare files with the same subject_id
            if subject_id_1 != subject_id_2:
                continue
            if (file1, file2) not in compared_pairs:
                # if the file content is too short, just ignore
                if os.path.getsize(file1) < 100 or os.path.getsize(file2) < 100:
                    continue
                if filecmp.cmp(file1, file2, shallow=False):
                    duplicates.append((os.path.basename(file1), os.path.basename(file2)))
                compared_pairs.add((file1, file2))
    
    duplicates_df = pd.DataFrame(duplicates, columns=['File1', 'File2'])
    duplicates_df.to_csv(output_file, index=False, encoding='utf-8')

with open('videos_shorter_than_2_second.txt', 'r') as f:
    short_videos = f.read().splitlines()
input_csv_folder = '/data/yinzi/babyview_20240507/transcripts_distil-large-v3/Babyview_Main'
csv_files = glob(input_csv_folder + '/**/*.csv', recursive=True)
csv_files_after_filtered = [file for file in csv_files if os.path.basename(file).replace('.csv', '') not in short_videos]

input_csv_folder = '/data/yinzi/babyview_20240507/transcripts_distil-large-v3/Babyview_Main'
output_txt_folder = './converted_txt_files'
output_duplicate_file = './duplicate_txt_files.csv'

csv_to_txt(csv_files_after_filtered, output_txt_folder)
find_duplicate_txt_files(output_txt_folder, output_duplicate_file)
