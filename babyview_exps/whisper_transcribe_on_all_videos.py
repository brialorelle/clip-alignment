import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from glob import glob
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract MP3 audio from video files using ffmpeg.")
    parser.add_argument("--mp3_folder", type=str, required=True, help="Folder to save extracted MP3 files.")
    parser.add_argument("--transcript_output_folder", type=str, required=True, help="Folder to save extracted transcripts.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    args = parser.parse_args()
    mp3_folder = args.mp3_folder
    transcript_output_folder = args.transcript_output_folder
    device = args.device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, 
                                                    torch_dtype=torch_dtype, 
                                                    low_cpu_mem_usage=True, 
                                                    use_safetensors=True,
                                                    attn_implementation="flash_attention_2")
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    all_audio_files = glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True)

    for audio_file in tqdm(all_audio_files):
        file_name = os.path.basename(audio_file)
        file_name = re.sub(r"\.mp3$", "", file_name)
        try:
            subject_id = re.findall(r'(.*)_GX', file_name)[0]
        except:
            print(f"Error: Could not extract subject ID from {file_name}, skip")
            continue
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
        output_path = os.path.join(transcript_output_folder, subject_id, f"{file_name}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index_label="utterance_no")

if __name__ == "__main__":
    main()

