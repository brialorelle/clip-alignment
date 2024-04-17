#%%
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

# Paths setup
babyview_video_folder = "/data/yinzi/babyview/Babyview_Main"
output_root_dir = "/data/yinzi/babyview/"
all_subject_number_list = sorted(os.listdir(babyview_video_folder))
# exclude pilot subjects
exclude_subjects = ['Bria_Long', 'Erica_Yoon']
all_subject_number_list = [subject for subject in all_subject_number_list if subject not in exclude_subjects]
total_number_of_videos = sum([len(glob(os.path.join(babyview_video_folder, subject, "*.MP4"))) for subject in all_subject_number_list])
topN = 250

# all_results = []  # This will store results from all subjects
utterances_max_dict = {}

for subject in tqdm(all_subject_number_list, desc="Processing Subjects"):
    all_mp4_files = glob(os.path.join(babyview_video_folder, subject, "*.MP4"))
    for mp4_full_path in tqdm(all_mp4_files, desc="Processing Videos"):
        video_file_name = os.path.splitext(os.path.basename(mp4_full_path))[0]
        # output_csv_dir = os.path.join(output_root_dir, "all_clip_results", subject, "all_result_csv_files", video_file_name, 'clip_final_results.csv')
        output_csv_dir = os.path.join(output_root_dir, "all_clip_results_large_v3", subject, "all_result_csv_files", video_file_name, 'clip_final_results.csv')
        
        # Checking if the result file exists
        if not os.path.exists(output_csv_dir):
            print(f"Missing data for {video_file_name}:{output_csv_dir}")
            continue
        
        single_video_df = pd.read_csv(output_csv_dir)
        single_video_df = single_video_df.dropna(subset=['dot_product'])  # skip NaN values
        single_video_df['utterance_no'] = single_video_df['utterance_no'].apply(lambda x: f"{video_file_name}_{x}")  # rename the utterance_no
        
        # Collecting data
        for _, group in single_video_df.groupby('utterance_no'):
            max_entry = group.loc[group['dot_product'].idxmax()]
            if max_entry['text'] not in utterances_max_dict:
                utterances_max_dict[max_entry['text']] = {
                    'max_dot_product': max_entry['dot_product'],
                    'max_frame': max_entry['frame'],
                    'utterance_no': max_entry['utterance_no'],
                    'text': max_entry['text']
                }
            else:
                if max_entry['dot_product'] > utterances_max_dict[max_entry['text']]['max_dot_product']:
                    utterances_max_dict[max_entry['text']] = {
                        'max_dot_product': max_entry['dot_product'],
                        'max_frame': max_entry['frame'],
                        'utterance_no': max_entry['utterance_no'],
                        'text': max_entry['text']
                    }


# Creating DataFrame from results
# all_results_df = pd.DataFrame(all_results)

# Sorting by 'max_dot_product' to find the top 200 utterances
# top_utterances_df = all_results_df.sort_values(by='max_dot_product', ascending=False).head(200)
max_utterances_df = pd.DataFrame(utterances_max_dict).T.reset_index()
topN_max_utterances_df = max_utterances_df.sort_values(by='max_dot_product', ascending=False).head(topN)

# %%
# Save these top utterances into a CSV
# save_frame_dir = os.path.join(output_root_dir, "all_clip_results", "chosen_frames_new")
save_frame_dir = os.path.join(output_root_dir, "all_clip_results_large_v3", "chosen_frames_large_v3")
os.makedirs(save_frame_dir, exist_ok=True)
topN_max_utterances_df.to_csv(os.path.join(save_frame_dir, "topN_utterances.csv"), index=False)

# Plotting and saving the frames for these top utterances
for index, row in tqdm(topN_max_utterances_df.iterrows(), total=topN_max_utterances_df.shape[0], desc="Adding Titles and Saving Frames"):
    # for frame_type in ['max_frame', 'min_frame']:
    for frame_type in ['max_frame']:
        frame_path = row[frame_type]
        frame_score = row[frame_type.replace('frame', 'dot_product')]
        dest_path = os.path.join(save_frame_dir, f"{frame_score:.5f}_{row['utterance_no']}_clip.jpg")
        
        img = Image.open(frame_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"{row['text']} ({frame_score:.2f})", fontsize=12)
        plt.axis('off')
        plt.savefig(dest_path, bbox_inches='tight', pad_inches=0)
        plt.close()

