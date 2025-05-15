import os
import cv2
import numpy as np
from closed_form_matting import process_video

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Define paths based on your project structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_dir = os.path.join(project_root, "Videos/input")
    output_dir = os.path.join(project_root, "Videos/output/matting/closed_form")
    
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    # Get all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_filename = f"closed_form_matting_{video_file}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing video: {video_file}")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        print("=" * 50)
        
        # Process the video
        success = process_video(input_path, output_path)
        
        if success:
            print(f"Completed processing {video_file}")
        else:
            print(f"Failed to process {video_file}")
        print("=" * 50)
    
    print("\nAll videos processed!")

if __name__ == "__main__":
    main()
