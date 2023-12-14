import os
import subprocess

# apply whisper to video
def apply_whisper(file, video_name, whisper_dir):
    # run whisper in terminal
    command = f"whisper {file} --model medium --language English --output_dir {whisper_dir}"
    print("command: ", command)
    # Use subprocess to execute the command
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    apply_whisper(file)