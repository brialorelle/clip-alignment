import os
import csv
import clip
import torch
import numpy as np

from PIL import Image
import argparse
args_parser = argparse.ArgumentParser()
args_parser.add_argument("--video_file", type=str, required=True)
args_parser.add_argument("--csv_file", type=str, required=True)
args_parser.add_argument("--device", type=str, default="cpu")
args_parser.add_argument("--save_root_dir", type=str, required=False)

args = args_parser.parse_args()
# Main script starts here
csv_file = args.csv_file
video_file = args.video_file
device = args.device
# name without extension
save_root_dir = args.save_root_dir
if save_root_dir is None:
    save_root_dir = "./"

video_file_name = os.path.splitext(os.path.basename(video_file))[0]

def save_embedding(embedding, dir_path, file_name):
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, file_name)
    np.save(path, embedding.detach().cpu().numpy())
    return path


def calculate_dot_product(frame_folder, text, utterance_no, model, token_limit=77):
    results = []
    # Split the text into chunks within the token limit
    text_chunks = [text[i:i+token_limit] for i in range(0, len(text), token_limit)]
    combined_text_features = None
    
    for chunk in text_chunks:
        # Preprocess and encode each text chunk
        text_input = clip.tokenize([chunk]).to(device)
        text_features_chunk = model.encode_text(text_input)
        text_features_chunk /= text_features_chunk.norm(dim=-1, keepdim=True)
        
        # Combine encoded features of all chunks by averaging
        if combined_text_features is None:
            combined_text_features = text_features_chunk
        else:
            combined_text_features += text_features_chunk
   
    # Average the combined text features if text was split into chunks
    if len(text_chunks) > 1:
        combined_text_features /= len(text_chunks)
    text_embedding_path = save_embedding(combined_text_features, os.path.join(save_root_dir, f'all_embeddings/text_embeddings/{video_file_name}'), f'{utterance_no}_text_embedding.npy')
    # Iterate through each frame in the folder
    for frame in os.listdir(frame_folder):
        if frame.endswith(".jpg"):
            frame_path = os.path.join(frame_folder, frame)
            # Load and preprocess the frame
            frame_image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
            
            # Encode the frame
            frame_features = model.encode_image(frame_image)
            frame_features /= frame_features.norm(dim=-1, keepdim=True)
            image_embedding_path = save_embedding(frame_features, os.path.join(save_root_dir, f'all_embeddings/image_embeddings/{video_file_name}/{utterance_no}'), f'{os.path.splitext(frame)[0]}_image_embedding.npy')

            # Calculate dot product
            dot_product = (text_embedding_path @ frame_features.T).detach().cpu().numpy()[0][0]
            del frame_image
            del frame_features
            torch.cuda.empty_cache()
            
            results.append({
                "utterance_no": utterance_no,
                "frame": frame_path,
                "text": text,
                "dot_product": dot_product,
                "text_embedding_path": text_embedding_path,
                "image_embedding_path": image_embedding_path
            })
    return results


# Load the CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
final_results = []
with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        output_dir =  os.path.join(save_root_dir, f'output_frames/{video_file_name}/{row["utterance_no"]}')
        os.makedirs(output_dir, exist_ok=True)
        results = calculate_dot_product(output_dir, row['text'], row['utterance_no'], model=model)
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
