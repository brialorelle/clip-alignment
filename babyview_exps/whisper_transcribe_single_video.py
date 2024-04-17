import os
import re
import torch
import argparse
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def main():

    parser = argparse.ArgumentParser(description="Extract MP3 audio from video files using ffmpeg.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--output_transcript_folder", type=str, required=True, help="Path to save extracted transcript.")
    args = parser.parse_args()
    device = args.device
    input_audio = args.input_audio
    output_transcript_folder = args.output_transcript_folder

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "distil-whisper/distil-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

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

    result = pipe(input_audio, return_timestamps=True)
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
    file_name = os.path.basename(input_audio)
    file_name = re.sub(r"\.mp3$", "", file_name)
    subject_id = file_name.split("_")[0]
    output_path = os.path.join(output_transcript_folder, subject_id, f"{file_name}.csv")
    res_df.to_csv(output_path, index_label="utterance_no")

if __name__ == "__main__":
    main()
