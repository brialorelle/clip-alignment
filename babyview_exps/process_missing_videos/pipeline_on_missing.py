#%% read in txt
import os
with open("missing.txt", "r") as f:
    missing_names = f.readlines()
    missing_names = [name.strip() for name in missing_names]
print(missing_names)
# %% preprocess names
video_root_path = "/data/yinzi/babyview_20240507/Babyview_Main"
csv_root_path = "/data/yinzi/babyview_20240507/transcripts_distil-large-v3/Babyview_Main"
save_root_dir = "/data/yinzi/babyview_20240507/all_clip_results_large_v3"
# extract subject id
subject_id = [name.split("_")[0] for name in missing_names]
print(subject_id)
video_path = [os.path.join(video_root_path, subject, name+".MP4") for subject, name in zip(subject_id, missing_names)]
print(video_path)
# check exit 
for path in video_path:
    if not os.path.exists(path):
        print(f"{path} not exists")
csv_path = [os.path.join(csv_root_path, subject, name+".csv") for subject, name in zip(subject_id, missing_names)]
print(csv_path)
for path in csv_path:
    if not os.path.exists(path):
        print(f"{path} not exists")
save_full_path = [os.path.join(save_root_dir, subject) for subject in subject_id]


#%% extract frames
import os
#  python extract_all_frames.py --video_file /data/yinzi/babyview_20240507/Babyview_Main/00220001/00220001_GX010001_02.05.2024-02.11.2024_02.05.2024-5:19pm.MP4 --csv_file /data/yinzi/babyview_20240507/transcripts_distil-large-v3/Babyview_Main/00220001/00220001_GX010001_02.05.2024-02.11.2024_02.05.2024-5:19pm.csv --save_root_dir /data/yinzi/babyview_20240507/all_clip_results_large_v3/00220001/
# extract_frame_output_path /data/yinzi/babyview_20240507/all_clip_results_large_v3/00220001/
print(f"Pipeline start")
print(f"1. Extracting frames from {len(video_path)} videos")
for video, csv, save in zip(video_path, csv_path, save_full_path):
    print(f"Extracting frames from {video}")
    os.system(f"python extract_all_frames.py --video_file {video} --csv_file {csv} --save_root_dir {save}")
# %%