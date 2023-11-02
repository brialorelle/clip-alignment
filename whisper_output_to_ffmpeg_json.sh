#!/bin/bash

input_video="example_clip.mov"
json_file="example_clip_mod.json"
output_directory="output_frames"

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
    
    ffmpeg -i "$input_video" -ss "$start_timestamp" -to "$stop_timestamp" -r 30 -f image2 "$output_directory/frame_%04d-new.jpg"
done

