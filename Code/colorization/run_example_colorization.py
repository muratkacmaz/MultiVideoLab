import os
import cv2
import numpy as np
from example_colorization import process_video, select_reference_image

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Define paths based on your project structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_dir = os.path.join(project_root, "Videos/input")
    output_dir = os.path.join(project_root, "Videos/output/colorization/example_based")
    reference_dir = os.path.join(project_root, "Videos/input")  # You might want a separate folder for reference images
    
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    # Ask user to select a reference image
    print("First, select a reference color image to use for colorization.")
    reference_path = select_reference_image()
    
    if reference_path is None:
        print("No reference image selected. Exiting.")
        return
    
    # Ask if user wants to use the advanced method
    use_advanced = input("Use advanced color transfer method? (y/n): ").lower() == 'y'
    
    # Get all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        method_name = "advanced" if use_advanced else "basic"
        output_filename = f"example_based_{method_name}_{video_file}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing video: {video_file}")
        print(f"Input path: {input_path}")
        print(f"Reference image: {reference_path}")
        print(f"Output path: {output_path}")
        
        # Call the process_video function
        success = process_video(input_path, reference_path, output_path, use_advanced)
        
        if success:
            print(f"Completed processing {video_file}")
        else:
            print(f"Failed to process {video_file}")
        print("=" * 50)

if __name__ == "__main__":
    main()
