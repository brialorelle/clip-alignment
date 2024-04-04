import os
import csv
import numpy as np
from PIL import Image
import argparse
from clip_client import Client
from docarray import Document, DocumentArray


# Initialize the argparse
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--video_file", type=str, required=True)
args_parser.add_argument("--csv_file", type=str, required=True)
args_parser.add_argument("--server_url", type=str, default="grpc://0.0.0.0:51000")
args_parser.add_argument("--save_root_dir", type=str, required=False)

args = args_parser.parse_args()

# Initialize the CLIP client
client = Client(args.server_url)

csv_file = args.csv_file
video_file = args.video_file
save_root_dir = args.save_root_dir or "./"

video_file_name = os.path.splitext(os.path.basename(video_file))[0]

def save_embedding(embedding, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, file_name)
    np.save(path, embedding)
    return path

def calculate_dot_product(frame_folder, text, utterance_no):
    results = []
    # Encode the text to get its embedding
    text_embedding = client.encode([text])[0]
    text_embedding /= np.linalg.norm(text_embedding)  # Normalization
    text_embedding_path = save_embedding(text_embedding, os.path.join(save_root_dir, f'all_embeddings/text_embeddings/{video_file_name}'), f'{utterance_no}_text_embedding.npy')
    
    for frame in os.listdir(frame_folder):
        if frame.endswith(".jpg"):
            frame_path = os.path.join(frame_folder, frame)
            # Encode the frame to get its embedding
            image_embedding = client.encode([frame_path])[0]
            image_embedding /= np.linalg.norm(image_embedding)  # Normalization

            image_embedding_path = save_embedding(image_embedding, os.path.join(save_root_dir, f'all_embeddings/image_embeddings/{video_file_name}/{utterance_no}'), f'{os.path.splitext(frame)[0]}_image_embedding.npy')
            
            # Calculate dot product
            dot_product = np.dot(text_embedding, image_embedding)
            
            results.append({
                "utterance_no": utterance_no,
                "frame": frame_path,
                "text": text,
                "dot_product": dot_product,
                "text_embedding_path": text_embedding_path,
                "image_embedding_path": image_embedding_path
            })
    return results

final_results = []
with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        output_dir = os.path.join(save_root_dir, f'output_frames/{video_file_name}/{row["utterance_no"]}')
        os.makedirs(output_dir, exist_ok=True)
        results = calculate_dot_product(output_dir, row['text'], row['utterance_no'])
        final_results.extend(results)

output_path = "all_result_csv_files"
output_dir = os.path.join(save_root_dir, output_path, video_file_name)
os.makedirs(output_dir, exist_ok=True)

# Write final results to a single CSV file
with open(os.path.join(output_dir,'clip_final_results.csv'), mode='w', newline='') as result_file:
    writer = csv.DictWriter(result_file, fieldnames=["utterance_no", "frame", "text", "dot_product", "text_embedding_path", "image_embedding_path"])
    writer.writeheader()
    for result in final_results:
        writer.writerow(result)
