import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from glob import glob
import pandas as pd
import os
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# Function to process each group of audio files
def process_audio_group(group_data):
    group_number, files, device, output_folder = group_data
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    
    # Setup the processor and model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2"
    )
    model.to(device)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device
    )

    for audio_file in files:
        file_name = os.path.basename(audio_file)
        file_name = re.sub(r"\.mp3$", "", file_name)
        subject_id = re.search(r'(.*)_GX', file_name)
        if not subject_id:
            print(f"Error: Could not extract subject ID from {file_name}, skipping")
            continue
        subject_id = subject_id.group(1)

        result = pipe(audio_file, return_timestamps=True)
        # Convert timestamps to continuous time
        current_time = 0.0
        data = []
        for chunk in result['chunks']:
            start_time = current_time
            end_time = start_time + (chunk['timestamp'][1] - chunk['timestamp'][0])
            data.append({
                'start_time': round(start_time, 2),
                'end_time': round(end_time, 2),
                'text': chunk['text']
            })
            current_time = end_time  # Update current time
        
        # Save result to csv
        res_df = pd.DataFrame(data)
        output_path = os.path.join(output_folder, subject_id, f"{file_name}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index_label="utterance_no")

def main():
    parser = argparse.ArgumentParser(description="Process audio files using Whisper.")
    parser.add_argument("--mp3_folder", type=str, required=True, help="Folder containing MP3 files.")
    parser.add_argument("--transcript_output_folder", type=str, required=True, help="Folder to save transcripts.")
    parser.add_argument("--device_ids", type=str, default="[0,1,2,3]", help="List of GPU device IDs to use.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of workers to use.")
    args = parser.parse_args()

    mp3_folder = args.mp3_folder
    transcript_output_folder = args.transcript_output_folder
    device_ids = [int(id) for id in args.device_ids.strip("[]").split(",")]
    num_devices = len(device_ids)

    all_audio_files = glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True)
    files_per_group = len(all_audio_files) // num_devices
    grouped_tasks = [
        (i, all_audio_files[i*files_per_group:(i+1)*files_per_group], f'cuda:{device_ids[i % num_devices]}', transcript_output_folder)
        for i in range(num_devices)
    ]
    # If there are leftovers, distribute them to the groups round-robin style
    leftovers = all_audio_files[files_per_group*num_devices:]
    for i, file in enumerate(leftovers):
        grouped_tasks[i % num_devices][1].append(file)

    # Process groups in parallel
    with ProcessPoolExecutor(max_workers=num_devices) as executor:
        list(tqdm(executor.map(process_audio_group, grouped_tasks), total=len(grouped_tasks)))

if __name__ == "__main__":
    main()
