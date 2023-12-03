import os
import subprocess

# apply whisper to video
def apply_whisper(file, video_name, whisper_dir):
    # run whisper in terminal
    command = f"whisper {file} --model medium --language English --output_dir {whisper_dir}"
    print("command: ", command)
    # Use subprocess to execute the command
    subprocess.run(command, shell=True)

    # get name of output file so the rest of the pipeline knows what to work with
    file_path = os.path.join(whisper_dir, video_name)
    file_path_2 = file_path + ".json"
    return file_path_2


if __name__ == "__main__":
    apply_whisper(file)