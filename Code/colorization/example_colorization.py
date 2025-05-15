import numpy as np
import cv2
import os

def rgb_to_lab(rgb_image):
    """
    Convert an RGB image to LAB color space.
    
    Parameters:
        rgb_image: Input RGB image
    
    Returns:
        Image in LAB color space
    """
    # OpenCV uses BGR, not RGB
    # Convert from BGR to LAB
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
    return lab_image

def lab_to_rgb(lab_image):
    """
    Convert a LAB image to RGB color space.
    
    Parameters:
        lab_image: Input LAB image
    
    Returns:
        Image in RGB color space (actually BGR for OpenCV)
    """
    # Convert from LAB to BGR
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return rgb_image

def match_color_statistics(source_lab, target_l_channel):
    """
    Match color statistics from source to target using Reinhard's method.
    
    Parameters:
        source_lab: Source image in LAB color space
        target_l_channel: Target grayscale image (L channel only)
    
    Returns:
        Colorized image in LAB color space
    """
    # Split the LAB image into its channels
    source_l, source_a, source_b = cv2.split(source_lab)
    
    # Compute statistics
    source_l_mean, source_l_std = np.mean(source_l), np.std(source_l)
    source_a_mean, source_a_std = np.mean(source_a), np.std(source_a)
    source_b_mean, source_b_std = np.mean(source_b), np.std(source_b)
    
    target_l_mean, target_l_std = np.mean(target_l_channel), np.std(target_l_channel)
    
    # Adjust the L channel of the target to match the statistics of the source
    target_l_adjusted = ((target_l_channel - target_l_mean) * (source_l_std / target_l_std)) + source_l_mean
    
    # Create new A and B channels based on source statistics
    height, width = target_l_channel.shape
    target_a = np.ones((height, width), dtype=np.float32) * source_a_mean
    target_b = np.ones((height, width), dtype=np.float32) * source_b_mean
    
    # Combine the channels
    colorized_lab = cv2.merge([target_l_adjusted.astype(np.uint8), target_a.astype(np.uint8), target_b.astype(np.uint8)])
    
    return colorized_lab

def colorize_image(target_gray, source_color):
    """
    Colorize a grayscale image using a reference color image.
    
    Parameters:
        target_gray: Grayscale image to be colorized
        source_color: Reference color image
    
    Returns:
        Colorized image in BGR format
    """
    # Convert the source color image to LAB
    source_lab = rgb_to_lab(source_color)
    
    # Convert grayscale to L channel
    if len(target_gray.shape) == 3:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
    
    # Match color statistics and transfer the color
    colorized_lab = match_color_statistics(source_lab, target_gray)
    
    # Convert back to RGB (BGR for OpenCV)
    colorized_rgb = lab_to_rgb(colorized_lab)
    
    return colorized_rgb

def advanced_color_transfer(target_gray, source_color):
    """
    A more advanced color transfer method that matches colors more accurately.
    
    Parameters:
        target_gray: Grayscale image to be colorized
        source_color: Reference color image
    
    Returns:
        Colorized image in BGR format
    """
    # Convert the source color image to LAB
    source_lab = rgb_to_lab(source_color)
    
    # Convert grayscale to L channel
    if len(target_gray.shape) == 3:
        target_gray = cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
    
    # Get dimensions
    height, width = target_gray.shape
    
    # Create an empty target LAB image
    target_lab = np.zeros((height, width, 3), dtype=np.uint8)
    target_lab[:,:,0] = target_gray  # L channel
    
    # Split the LAB image into its channels
    source_l, source_a, source_b = cv2.split(source_lab)
    
    # Match histograms for better color transfer
    target_lab[:,:,1] = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(source_a.astype(np.uint8))
    target_lab[:,:,2] = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(source_b.astype(np.uint8))
    
    # Convert back to RGB (BGR for OpenCV)
    colorized_rgb = lab_to_rgb(target_lab)
    
    return colorized_rgb

def process_video(input_path, reference_path, output_path, use_advanced=False):
    """
    Process a video by colorizing each frame using an example-based method.
    
    Parameters:
        input_path: Path to the input grayscale video
        reference_path: Path to the reference color image
        output_path: Path to save the colorized video
        use_advanced: Whether to use the advanced color transfer method
        
    Returns:
        True if successful, False otherwise
    """
    # Check if input and reference exist
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
        return False
    
    if not os.path.exists(reference_path):
        print(f"Error: Reference image not found at {reference_path}")
        return False
    
    # Load reference color image
    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        print(f"Error: Could not read reference image from {reference_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resize reference image to match video frame size
    reference_image = cv2.resize(reference_image, (width, height))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
    
    # Process each frame
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale if not already
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Colorize the frame
        if use_advanced:
            colorized = advanced_color_transfer(gray, reference_image)
        else:
            colorized = colorize_image(gray, reference_image)
        
        # Write the frame
        out.write(colorized)
        
        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == frame_count:
            print(f"Processed frame {frame_idx}/{frame_count}")
    
    # Release resources
    cap.release()
    out.release()
    print("Video colorization complete!")
    return True

def select_reference_image():
    """
    Interactive UI for selecting a reference color image.
    
    Returns:
        Path to the selected reference image, or None if cancelled
    """
    # This would typically be implemented with a file dialog
    # For this example, we'll use a simple command-line input
    print("\nPlease enter the path to the reference color image:")
    reference_path = input("> ")
    
    if not os.path.exists(reference_path):
        print(f"Error: File not found at {reference_path}")
        return None
    
    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        print(f"Error: Could not read image from {reference_path}")
        return None
    
    # Display the selected reference image
    cv2.namedWindow("Selected Reference Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Selected Reference Image", reference_image)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()
    
    return reference_path
