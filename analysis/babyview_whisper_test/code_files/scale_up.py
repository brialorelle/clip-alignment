import os
import pandas as pd

def scaling():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    print(script_directory)
    #image_dir = os.path.join(script_directory, 'output_frames_decimate')

    #csv_path = os.path.join(script_directory, 'output.csv')
    #items = pd.read_csv(csv_path)

    # Find the last index of '/'
    last_slash_index = script_directory.rfind('/')

    # Remove everything after the last '/'
    dir = script_directory[:last_slash_index + 1]

    print(dir)

    






if __name__ == "__main__":
    scaling()