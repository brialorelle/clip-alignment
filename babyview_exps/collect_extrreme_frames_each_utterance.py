# %% Sample the highest and lowest score frames for each subject
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

babyview_video_folder = "/data/yinzi/babyview/Babyview_Main"
output_root_dir = "/data/yinzi/babyview/"
all_subject_number_list = sorted(os.listdir(babyview_video_folder))
# exclude pilot subjects
exlucde_subjects = ['Bria_Long', 'Erica_Yoon']
all_subject_number_list = [subject for subject in all_subject_number_list if subject not in exlucde_subjects]
total_number_of_videos = sum([len(glob(os.path.join(babyview_video_folder, subject, "*.MP4")) ) for subject in all_subject_number_list])


all_chosen_frames_df = pd.DataFrame()
for subject in tqdm(all_subject_number_list):
    all_mp4_files = glob(os.path.join(babyview_video_folder, subject, "*.MP4"))
    results = []
    for mp4_full_path in tqdm(all_mp4_files):
        video_file_name = os.path.splitext(os.path.basename(mp4_full_path))[0]
        output_csv_dir = os.path.join(output_root_dir, "all_clip_results",subject, "all_result_csv_files", video_file_name,'clip_final_results.csv')
        single_video_df = pd.read_csv(output_csv_dir)
        # skip Nan values
        single_video_df = single_video_df.dropna(subset=['dot_product'])
        # rename the utterance_no in format: {video_name}_{utterance_no}
        single_video_df['utterance_no'] = single_video_df['utterance_no'].apply(lambda x: f"{video_file_name}_{x}")
        utterances = single_video_df['utterance_no'].unique()
        for utterance_no in utterances:
            utterance_df = single_video_df[single_video_df['utterance_no'] == utterance_no]
            utterance_transcript = utterance_df['text'].iloc[0]
            max_frame = utterance_df.loc[utterance_df['dot_product'].idxmax()]
            min_frame = utterance_df.loc[utterance_df['dot_product'].idxmin()]
            max_dot_product = max_frame['dot_product']
            min_dot_product = min_frame['dot_product']
            if max_dot_product == min_dot_product:
                # print(f"Same dot product for {utterance_no}")
                continue
            result = {
                "utterance_no": utterance_no,
                "text": utterance_transcript,
                "max_frame": max_frame['frame'],
                "max_dot_product": max_dot_product,
                "min_frame": min_frame['frame'],
                "min_dot_product": min_dot_product
            }
            results.append(result)
    extreme_frame_df = pd.DataFrame(results)
    sample_number = 10
    if len(extreme_frame_df) < sample_number:
        print(f"Less than 10 valid utterances for {subject}, only {len(extreme_frame_df)} valid utterances.")
        sample_number = len(extreme_frame_df)
        print(len(utterances))
        print(len(all_mp4_files))
        print(output_csv_dir)
    # random sample some utterances for each subject
    chosen_frame_df = extreme_frame_df.sample(sample_number, random_state=1)
    all_chosen_frames_df = pd.concat([all_chosen_frames_df, chosen_frame_df], ignore_index=True)
# %% Save the chosen frames to a folder
save_frame_dir = os.path.join(output_root_dir, "all_clip_results","chosen_frames")
os.makedirs(save_frame_dir, exist_ok=True)
# Plotting and saving frames with titles
for index, row in tqdm(all_chosen_frames_df.iterrows(), total=all_chosen_frames_df.shape[0], desc="Adding Titles"):
    for frame_type in ['max_frame', 'min_frame']:
        frame_path = row[frame_type]
        frame_score = row[frame_type.replace('frame', 'dot_product')]
        dest_path = os.path.join(save_frame_dir, f"{row['utterance_no']}_clip_{frame_score:.5f}.jpg")
        
        # Open the image file
        img = Image.open(frame_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(row['text'], fontsize=12)  # Set the transcript text as the title
        plt.axis('off')  # Hide axes
        plt.savefig(dest_path, bbox_inches='tight', pad_inches=0)
        plt.close()
# save all_chosen_frames_df to a csv file
all_chosen_frames_df.to_csv(os.path.join(save_frame_dir, "chosen_frames.csv"), index=False)