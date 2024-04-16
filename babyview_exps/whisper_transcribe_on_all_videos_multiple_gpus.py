import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from glob import glob
import pandas as pd
import os
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import cycle

# Function to process each audio file
def process_audio(args):
    audio_file, device, output_folder = args
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v3"
    processor = AutoProcessor.from_pretrained(model_id)
    if device in model_device_map_dict:
        model, pipe = model_device_map_dict[device]
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id,
                                                        torch_dtype=torch_dtype,
                                                        low_cpu_mem_usage=True,
                                                        use_safetensors=True,
                                                        attn_implementation="flash_attention_2")
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
        model_device_map_dict[device] = (model, pipe)

    file_name = os.path.basename(audio_file)
    file_name = re.sub(r"\.mp3$", "", file_name)
    subject_id = re.search(r'(.*)_GX', file_name)
    if not subject_id:
        print(f"Error: Could not extract subject ID from {file_name}, skipping")
        return
    subject_id = subject_id.group(1)

    result = pipe(audio_file, return_timestamps=True)
    res_df = pd.DataFrame({
        'start_time': [c['timestamp'][0] for c in result['chunks']],
        'end_time': [c['timestamp'][1] for c in result['chunks']],
        'text': [c['text'] for c in result['chunks']]
    })

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

    # only initialize the model once per device
    global model_device_map_dict
    model_device_map_dict = {}

    mp3_folder = args.mp3_folder
    transcript_output_folder = args.transcript_output_folder
    device_ids = [int(id) for id in args.device_ids.strip("[]").split(",")]

    all_audio_files = glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True)
    tasks = [(file, f'cuda:{device_id}', transcript_output_folder) for file in all_audio_files for device_id in device_ids]

    max_workers = args.max_workers
    # with ProcessPoolExecutor(max_workers=len(device_ids)) as executor:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_audio, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()