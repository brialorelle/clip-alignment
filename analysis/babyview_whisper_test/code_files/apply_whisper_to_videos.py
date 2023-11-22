import subprocess

def apply_whisper(file):
    # Your loop
    #for i in range(5):
        # Formulate your command based on the loop variable or any other conditions

    parts = file.split('/')
    babyview_whisper_test_list = parts[:-2]
    # Join the modified parts back into a string
    babyview_whisper_test = "/".join(babyview_whisper_test_list)
    whisper_output = babyview_whisper_test + "/whisper_output/"
    print(whisper_output)

    command = f"whisper {file} --model medium --language English --output_dir {whisper_output}"

        # Use subprocess to execute the command
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    apply_whisper("/Users/marielleib/Documents/GitHub/clip-alignment/analysis/babyview_whisper_test/videos/example_clip.mov")