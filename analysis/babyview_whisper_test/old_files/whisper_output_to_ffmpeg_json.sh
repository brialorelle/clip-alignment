#!/bin/bash

# NOT USING  - repeat with extract_frames.py I think

# eg. output.mp4
input_video="$1"
# eg. output_mod.json
json_file="$2"
# eg. output_frames_decimate
output_directory="$3"

# Check if jq is installed
if ! command -v jq &>/dev/null; then
    echo "jq is not installed. Please install it."
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg is not installed. Please install it."
    exit 1
fi

# Parse JSON and extract frames
jq -c '.[]' "$json_file" | while read -r segment; do

    echo "Segment: $segment"

    start_timestamp=$(echo "$segment" | jq -r '.start')
    stop_timestamp=$(echo "$segment" | jq -r '.end')
    
    echo "Start Timestamp: $start_timestamp"
    echo "Stop Timestamp: $stop_timestamp"
    
    ffmpeg -i "$input_video" -ss "$start_timestamp" -to "$stop_timestamp" -r 30 -f image2 "$output_directory/$input_video/frame_%04d-new.jpg"
done

