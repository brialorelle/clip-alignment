#%%
import os
from glob import glob
from tqdm import tqdm
# Paths setup
babyview_video_folder = "/data/yinzi/babyview/Babyview_Main"
output_root_dir = "/data/yinzi/babyview/"
all_subject_number_list = sorted(os.listdir(babyview_video_folder))
# exclude pilot subjects
exclude_subjects = ['Bria_Long', 'Erica_Yoon']
all_subject_number_list = [subject for subject in all_subject_number_list if subject not in exclude_subjects]
# use ffmpeg to get the duration of each video
total_duration = 0
for subject in tqdm(all_subject_number_list, desc="Processing Subjects"):
    all_mp4_files = glob(os.path.join(babyview_video_folder, subject, "*.MP4"))
    for mp4_full_path in all_mp4_files:
        cmd = f"ffprobe -v error -ignore_chapters 1 -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {mp4_full_path}"
        duration = float(os.popen(cmd).read())
        total_duration += duration
# format the total duration in hours
total_duration = total_duration / 3600
print(f"Total duration of all videos: {total_duration} hours")