import os
import csv
import torch
import numpy as np
import argparse
from clip_client import Client
from docarray import Document, DocumentArray

# Initialize the argparse
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--video_file", type=str, required=True)
args_parser.add_argument("--csv_file", type=str, required=True)
args_parser.add_argument("--server_url", type=str, default="grpc://0.0.0.0:51000")
args_parser.add_argument("--save_root_dir", type=str, required=False)
args_parser.add_argument("--frame_batch_size", type=int, default=512)
args_parser.add_argument("--clip_batch_size", type=int, default=256)
args_parser.add_argument("--prefetch", type=int, default=100)

args = args_parser.parse_args()

# Ensure that torch is using the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the CLIP client
client = Client(args.server_url)

# Setup directories
csv_file = args.csv_file
video_file = args.video_file
save_root_dir = args.save_root_dir or "./"
frame_batch_size = args.frame_batch_size
clip_batch_size = args.clip_batch_size
prefetch = args.prefetch

video_file_name = os.path.splitext(os.path.basename(video_file))[0]

print("Starting batch processing for all frames of video:", video_file_name)

def save_embedding(embedding, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, file_name)
    # Convert tensor to numpy before saving
    np_embedding = embedding.cpu().numpy()
    np.save(path, np_embedding)
    return path

# Collect unique texts
texts = set()
utterances_info = []
with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        if row['speaker'] not in ['FEM', 'MAL']:
            print(f"[{os.path.basename(csv_file)}]:\nSkipping non-adult utterance<utterance_no,speaker>: <{row['utterance_no'],row['speaker']}>")
            continue
        texts.add(row['text'])
        utterances_info.append((row['utterance_no'], row['text']))

# Convert set back to list for ordering
texts = list(texts)
# remove empty strings
texts = [text for text in texts if text]
# Get all unique text embeddings at once
print(f"Processing texts, clip_batch_size={clip_batch_size}...")
text_embeddings = torch.tensor(client.encode(texts, batch_size=clip_batch_size, show_progress=True, prefetch=prefetch)).to(device)
text_embeddings /= torch.norm(text_embeddings, dim=1, keepdim=True)

# Create a mapping of text to its embedding
text_to_embedding = {text: embedding for text, embedding in zip(texts, text_embeddings)}

# Collect all frame paths
frame_paths = []
for utterance_no, text in utterances_info:
    output_dir = os.path.join(save_root_dir, f'output_frames/{video_file_name}/{utterance_no}')
    os.makedirs(output_dir, exist_ok=True)
    frame_paths.extend([(utterance_no, text, os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith(".jpg")])

if(len(frame_paths) == 0):
    raise ValueError(f"No frames found for the video:{frame_paths}")
print(f"Processing frames in batches, frame_batch_size={frame_batch_size}, clip_batch_size={clip_batch_size}...")
# Process frame paths in batches and calculate dot products
results = []
for i in range(0, len(frame_paths), frame_batch_size):
    batch_frame_paths = frame_paths[i:i+frame_batch_size]
    paths = [path for _, _, path in batch_frame_paths]
    if not os.path.exists(paths[0]):
        raise ValueError(f"Frame path not found : {paths[0]}")
    batch_image_embeddings = torch.tensor(client.encode(paths, batch_size=clip_batch_size, show_progress=True, prefetch=prefetch)).to(device)
    batch_image_embeddings /= torch.norm(batch_image_embeddings, dim=1, keepdim=True)

    for j, (utterance_no, text, frame_path) in enumerate(batch_frame_paths):
        if not text:
            continue
        text_embedding = text_to_embedding[text]
        image_embedding = batch_image_embeddings[j]
        dot_product = torch.dot(text_embedding, image_embedding).item()
        
        text_embedding_path = save_embedding(text_embedding, os.path.join(save_root_dir, f'all_embeddings/text_embeddings/{video_file_name}'), f'{utterance_no}_text_embedding.npy')
        image_embedding_path = save_embedding(image_embedding, os.path.join(save_root_dir, f'all_embeddings/image_embeddings/{video_file_name}/{utterance_no}'), f'{os.path.splitext(os.path.basename(frame_path))[0]}_image_embedding.npy')
        
        results.append({
            "utterance_no": utterance_no,
            "frame": frame_path,
            "text": text,
            "dot_product": dot_product,
            "text_embedding_path": text_embedding_path,
            "image_embedding_path": image_embedding_path
        })

# Save results
output_dir = os.path.join(save_root_dir, "all_result_csv_files", video_file_name)
print(output_dir)
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'clip_final_results.csv'), mode='w', newline='') as result_file:
    writer = csv.DictWriter(result_file, fieldnames=["utterance_no", "frame", "text", "dot_product", "text_embedding_path", "image_embedding_path"])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Video {video_file_name} processing completed!")