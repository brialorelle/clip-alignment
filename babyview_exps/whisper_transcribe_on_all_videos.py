import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from glob import glob
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Extract MP3 audio from video files using ffmpeg.")
    parser.add_argument("--mp3_folder", type=str, required=True, help="Folder to save extracted MP3 files.")
    parser.add_argument("--transcript_output_folder", type=str, required=True, help="Folder to save extracted transcripts.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--rank_id", type=int, default=0, help="Rank ID for distributed running.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--is_saycam", type=int, default=0, help="Whether the videos are from SayCam.")
    args = parser.parse_args()
    mp3_folder = args.mp3_folder
    is_saycam = args.is_saycam
    transcript_output_folder = args.transcript_output_folder
    device = args.device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v3"
    rank_id = args.rank_id
    num_parallel = args.num_parallel
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, 
                                                    torch_dtype=torch_dtype, 
                                                    low_cpu_mem_usage=True, 
                                                    use_safetensors=True,
                                                    attn_implementation="flash_attention_2")
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=25,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    all_audio_files = sorted(glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True))

    group_size = len(all_audio_files) // num_parallel
    start_idx = rank_id * group_size
    end_idx = start_idx + group_size
    if rank_id == num_parallel - 1:
        end_idx = len(all_audio_files)
    current_group_audio_files = all_audio_files[start_idx:end_idx]

    for idx, audio_file in enumerate(tqdm(current_group_audio_files)):
        file_name = os.path.basename(audio_file)
        file_name = re.sub(r"\.mp3$", "", file_name)
        try:
            # fix it because the partten is different for luna vidoes
            partten = re.findall(r'(.*)_GX', file_name)
            if len(partten) == 0: # new name partten
                partten = re.findall(r'(.*)_H', file_name)
            if len(partten) == 0:  # fallback partten
                partten = file_name.split("_")
            subject_id = partten[0]
            # subject_id = file_name.split("_")[0]
        except:
            print(f"[WARNING]: Could not extract subject ID from {file_name}, skip")
            raise RuntimeError
        result = pipe(audio_file, return_timestamps=True)
        # current_time = 0.0
        data = []
        for chunk in result['chunks']:
            start_time = chunk['timestamp'][0]
            end_time = chunk['timestamp'][1]
            # end_time = start_time + (duration_end - duration_start)
            data.append({
                'start_time': start_time,
                'end_time': end_time,
                'text': chunk['text']
            })
            # current_time = end_time  # Update current time
        # Save result to csv
        res_df = pd.DataFrame(data)
        output_path = os.path.join(transcript_output_folder, subject_id, f"{file_name}.csv")
        if is_saycam:
            output_path = os.path.join(transcript_output_folder, f"{file_name}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index_label="utterance_no")

if __name__ == "__main__":
    main()

