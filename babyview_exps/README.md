# Installation

```
conda create --name torch python=3.10
conda activate torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ffmpeg
pip install numpy pandas tqdm opencv-python pillow
pip install clip-server
pip install clip-client
pip install nvidia-pyindex 
pip install "clip_server[tensorrt]"
```

# Start CLIP Server(TensorRT accelerated)
```
cd ~/workspace/clip-alignment/babyview_exps
conda activate torch
python -m clip_server ./tensorrt-flow_replica_4.yml
```

# Run CLIP on all videos(Batch mode)
```
python all_videos_clip_batch.py --babyview_folder /data/yinzi/babyview/Babyview_Main --babyview_transcript_folder /data/yinzi/babyview/transcripts/Babyview_Main  --output_root_dir /data/yinzi/babyview/all_clip_results 
```
