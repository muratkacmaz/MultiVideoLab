import os
import cv2
import numpy as np
from scribble_colorization import colorize_image, process_video

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Define paths based on your project structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_dir = os.path.join(project_root, "Videos/input")
    output_dir = os.path.join(project_root, "Videos/output/colorization/scribble_based")
    
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    # Get all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_filename = f"scribble_based_{video_file}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing video: {video_file}")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        
        # Call the process_video function
        success = process_video(input_path, output_path)
        
        if success:
            print(f"Completed processing {video_file}")
        else:
            print(f"Skipped or failed processing {video_file}")
        print("=" * 50)

if __name__ == "__main__":
    main()
