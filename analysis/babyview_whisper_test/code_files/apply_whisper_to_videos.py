import os
import subprocess

def apply_whisper(file):
    parts = file.split('/')
    filename = parts[-1].split('.')[0]
    babyview_whisper_test_list = parts[:-2]
    # Join the modified parts back into a string
    babyview_whisper_test = "/".join(babyview_whisper_test_list)
    whisper_output = os.path.join(babyview_whisper_test, "whisper_output")
    print("folder for whisper output: ", whisper_output)

    command = f"whisper {file} --model medium --language English --output_dir {whisper_output}"

    print("command: ", command)
    # Use subprocess to execute the command
    subprocess.run(command, shell=True)

    file_path = os.path.join(whisper_output, filename)
    file_path_2 = file_path + ".json"
    return file_path_2


if __name__ == "__main__":
    apply_whisper(file)