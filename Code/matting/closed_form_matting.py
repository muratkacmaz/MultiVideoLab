
import numpy as np
import cv2
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import os

def closed_form_matting(image, scribbles, lambda_value=100.0):
    """
    Implements the closed-form solution to natural image matting.
    
    Parameters:
        image: Input RGB image
        scribbles: Image with scribbles. Pixels marked with blue (0,0,255) are background, 
                   pixels marked with red (255,0,0) are foreground
        lambda_value: Weighting factor for the scribble constraints
    
    Returns:
        Alpha matte (0 = background, 1 = foreground)
    """
    print("Starting closed-form matting...")
    
    # Get dimensions
    height, width, channels = image.shape
    n_pixels = height * width
    
    # Normalize image to [0, 1]
    image_float = image.astype(np.float64) /.255
    
    # Create index matrix for easier reference
    indices = np.arange(n_pixels).reshape(height, width)
    
    # Generate foreground and background constraints from scribbles
    fg_mask = np.zeros((height, width), dtype=np.bool_)
    bg_mask = np.zeros((height, width), dtype=np.bool_)
    
    # Red pixels (BGR format in OpenCV) are foreground
    fg_mask = (scribbles[:,:,2] > 250) & (scribbles[:,:,0] < 10) & (scribbles[:,:,1] < 10)
    
    # Blue pixels (BGR format in OpenCV) are background
    bg_mask = (scribbles[:,:,0] > 250) & (scribbles[:,:,1] < 10) & (scribbles[:,:,2] < 10)
    
    print(f"Found {np.sum(fg_mask)} foreground scribble pixels and {np.sum(bg_mask)} background scribble pixels")
    
    # If no scribbles are found, return an error
    if np.sum(fg_mask) == 0 or np.sum(bg_mask) == 0:
        print("Error: No foreground or background scribbles found")
        return None
    
    # Create window size parameters (typically a 3x3 window)
    win_size = 1
    
    # Compute sparse matrix L (Laplacian matrix)
    print("Computing Laplacian matrix...")
    L_data = []
    L_rows = []
    L_cols = []
    
    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            i = indices[y, x]
            
            # For each pixel, consider a window around it
            window_indices = []
            window_pixels = []
            
            for wy in range(max(0, y - win_size), min(height, y + win_size + 1)):
                for wx in range(max(0, x - win_size), min(width, x + win_size + 1)):
                    if wy != y or wx != x:  # Exclude the center pixel
                        window_indices.append(indices[wy, wx])
                        window_pixels.append(image_float[wy, wx])
            
            if not window_pixels:
                continue
            
            # Convert to numpy arrays
            window_pixels = np.array(window_pixels)
            
            # Compute mean and covariance of the window
            mean = np.mean(window_pixels, axis=0)
            
            # Add regularization to make sure covariance matrix is well-conditioned
            cov = np.cov(window_pixels.T) + 0.00001 * np.eye(3)
            
            # Compute weights for the window pixels
            inv_cov = np.linalg.inv(cov)
            weights = []
            central_pixel = image_float[y, x]
            
            for pixel in window_pixels:
                diff = pixel - mean
                weight = 1.0 + (central_pixel - mean) @ inv_cov @ diff.T
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights)
            weights /= np.sum(weights)
            
            # Add diagonal element (center pixel)
            L_rows.append(i)
            L_cols.append(i)
            L_data.append(1.0)
            
            # Add off-diagonal elements (window pixels)
            for j, idx in enumerate(window_indices):
                L_rows.append(i)
                L_cols.append(idx)
                L_data.append(-weights[j])
    
    # Create the Laplacian matrix
    L = sparse.csr_matrix((L_data, (L_rows, L_cols)), shape=(n_pixels, n_pixels))
    
    # Create the constraint matrices
    print("Setting up constraint matrices...")
    D = sparse.lil_matrix((n_pixels, n_pixels))
    b = np.zeros(n_pixels)
    
    # Add foreground constraints (alpha = 1)
    for y in range(height):
        for x in range(width):
            if fg_mask[y, x]:
                idx = indices[y, x]
                D[idx, idx] = lambda_value
                b[idx] = lambda_value
    
    # Add background constraints (alpha = 0)
    for y in range(height):
        for x in range(width):
            if bg_mask[y, x]:
                idx = indices[y, x]
                D[idx, idx] = lambda_value
                b[idx] = 0.0
    
    # Convert D to CSR format for efficient operations
    D = D.tocsr()
    
    # Solve the linear system (L + D) * alpha = b
    print("Solving linear system...")
    A = L + D
    alpha = spsolve(A, b)
    
    # Reshape and clip the result
    alpha = alpha.reshape(height, width)
    alpha = np.clip(alpha, 0, 1)
    
    print("Matting completed!")
    return alpha

def draw_scribbles(image, window_name="Draw Scribbles"):
    """
    Interactive UI for drawing foreground and background scribbles.
    
    Parameters:
        image: Input image
        window_name: Name of the window
    
    Returns:
        Scribbled image with foreground (red) and background (blue) markers
    """
    # Create a copy for drawing scribbles
    scribbled_image = image.copy()
    
    # Current drawing color and state
    current_color = (0, 0, 255)  # Start with blue (BGR) for background
    brush_size = 5
    drawing = False
    last_point = None
    
    # Window setup
    cv2.namedWindow(window_name)
    
    # Function to handle mouse events
    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, last_point
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(scribbled_image, last_point, (x, y), current_color, brush_size)
                last_point = (x, y)
                cv2.imshow(window_name, scribbled_image)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    # Set the mouse callback
    cv2.setMouseCallback(window_name, on_mouse)
    
    print("\nInstructions:")
    print("- Draw scribbles with the left mouse button")
    print("- Press 'f' for foreground (red)")
    print("- Press 'b' for background (blue)")
    print("- Press '+' to increase brush size, '-' to decrease")
    print("- Press 'c' when done drawing")
    print("- Press 'r' to reset all scribbles")
    print("- Press 'q' to cancel\n")
    
    # Main interaction loop
    while True:
        cv2.imshow(window_name, scribbled_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('f'):
            current_color = (0, 0, 255)  # Red in BGR
            print("Selected: Foreground (Red)")
        elif key == ord('b'):
            current_color = (255, 0, 0)  # Blue in BGR
            print("Selected: Background (Blue)")
        elif key == ord('+'):
            brush_size = min(30, brush_size + 1)
            print(f"Brush size: {brush_size}")
        elif key == ord('-'):
            brush_size = max(1, brush_size - 1)
            print(f"Brush size: {brush_size}")
        elif key == ord('r'):
            scribbled_image = image.copy()
            print("Reset all scribbles")
        elif key == ord('c'):
            print("Finished drawing scribbles")
            break
        elif key == ord('q'):
            print("Cancelled")
            return None
    
    cv2.destroyWindow(window_name)
    return scribbled_image

def process_image(image_path, output_alpha_path=None, output_composite_path=None):
    """
    Process a single image using the closed-form matting algorithm.
    
    Parameters:
        image_path: Path to the input image
        output_alpha_path: Path to save the alpha matte
        output_composite_path: Path to save the composite result
    
    Returns:
        alpha_matte: The computed alpha matte
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Draw scribbles
    scribbled_image = draw_scribbles(image, "Draw Scribbles for Matting")
    if scribbled_image is None:
        return None
    
    # Compute alpha matte
    alpha_matte = closed_form_matting(image, scribbled_image)
    if alpha_matte is None:
        return None
    
    # Save alpha matte if requested
    if output_alpha_path:
        cv2.imwrite(output_alpha_path, (alpha_matte * 255).astype(np.uint8))
        print(f"Alpha matte saved to {output_alpha_path}")
    
    # Create and save composite if requested
    if output_composite_path:
        # Create a green screen background
        green_bg = np.zeros_like(image)
        green_bg[:,:] = (0, 255, 0)  # Green in BGR
        
        # Composite the foreground over the green background
        alpha_3channel = np.stack([alpha_matte] * 3, axis=2)
        composite = (image * alpha_3channel + green_bg * (1 - alpha_3channel)).astype(np.uint8)
        
        cv2.imwrite(output_composite_path, composite)
        print(f"Composite saved to {output_composite_path}")
    
    return alpha_matte

def process_frame(frame, fg_mask, bg_mask, lambda_value=100.0):
    """
    Process a video frame using the closed-form matting algorithm with existing scribble masks.
    
    Parameters:
        frame: Input video frame
        fg_mask: Binary mask of foreground scribbles
        bg_mask: Binary mask of background scribbles
        lambda_value: Weighting factor for the scribble constraints
    
    Returns:
        Alpha matte for the frame
    """
    # Get dimensions
    height, width, channels = frame.shape
    n_pixels = height * width
    
    # Normalize image to [0, 1]
    frame_float = frame.astype(np.float64) / 255.0
    
    # Create index matrix for easier reference
    indices = np.arange(n_pixels).reshape(height, width)
    
    # Create window size parameters
    win_size = 1
    
    # Compute sparse matrix L (Laplacian matrix)
    L_data = []
    L_rows = []
    L_cols = []
    
    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            i = indices[y, x]
            
            # For each pixel, consider a window around it
            window_indices = []
            window_pixels = []
            
            for wy in range(max(0, y - win_size), min(height, y + win_size + 1)):
                for wx in range(max(0, x - win_size), min(width, x + win_size + 1)):
                    if wy != y or wx != x:  # Exclude the center pixel
                        window_indices.append(indices[wy, wx])
                        window_pixels.append(frame_float[wy, wx])
            
            if not window_pixels:
                continue
            
            # Convert to numpy arrays
            window_pixels = np.array(window_pixels)
            
            # Compute mean and covariance of the window
            mean = np.mean(window_pixels, axis=0)
            
            # Add regularization
            cov = np.cov(window_pixels.T) + 0.00001 * np.eye(3)
            
            # Compute weights for the window pixels
            inv_cov = np.linalg.inv(cov)
            weights = []
            central_pixel = frame_float[y, x]
            
            for pixel in window_pixels:
                diff = pixel - mean
                weight = 1.0 + (central_pixel - mean) @ inv_cov @ diff.T
                weights.append(weight)
            
            # Normalize weights
            weights = np.array(weights)
            weights = np.abs(weights)  # Ensure weights are positive
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
            
            # Add diagonal element (center pixel)
            L_rows.append(i)
            L_cols.append(i)
            L_data.append(1.0)
            
            # Add off-diagonal elements (window pixels)
            for j, idx in enumerate(window_indices):
                L_rows.append(i)
                L_cols.append(idx)
                L_data.append(-weights[j])
    
    # Create the Laplacian matrix
    L = sparse.csr_matrix((L_data, (L_rows, L_cols)), shape=(n_pixels, n_pixels))
    
    # Create the constraint matrices
    D = sparse.lil_matrix((n_pixels, n_pixels))
    b = np.zeros(n_pixels)
    
    # Add foreground constraints (alpha = 1)
    for y in range(height):
        for x in range(width):
            if fg_mask[y, x]:
                idx = indices[y, x]
                D[idx, idx] = lambda_value
                b[idx] = lambda_value
    
    # Add background constraints (alpha = 0)
    for y in range(height):
        for x in range(width):
            if bg_mask[y, x]:
                idx = indices[y, x]
                D[idx, idx] = lambda_value
                b[idx] = 0.0
    
    # Convert D to CSR format for efficient operations
    D = D.tocsr()
    
    # Solve the linear system (L + D) * alpha = b
    A = L + D
    alpha = spsolve(A, b)
    
    # Reshape and clip the result
    alpha = alpha.reshape(height, width)
    alpha = np.clip(alpha, 0, 1)
    
    return alpha

def process_video(input_path, output_path):
    """
    Process a video using the closed-form matting algorithm.
    
    Parameters:
        input_path: Path to the input video
        output_path: Path to save the output video
    
    Returns:
        True if successful, False otherwise
    """
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
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
    
    # Read the first frame for drawing scribbles
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return False
    
    # Draw scribbles on the first frame
    print("Please draw foreground (red) and background (blue) scribbles on the first frame")
    scribbled_frame = draw_scribbles(first_frame, "Draw Scribbles for Video Matting")
    if scribbled_frame is None:
        return False
    
    # Extract foreground and background masks from scribbles
    fg_mask = (scribbled_frame[:,:,2] > 250) & (scribbled_frame[:,:,0] < 10) & (scribbled_frame[:,:,1] < 10)
    bg_mask = (scribbled_frame[:,:,0] > 250) & (scribbled_frame[:,:,1] < 10) & (scribbled_frame[:,:,2] < 10)
    
    # Check if masks are valid
    if np.sum(fg_mask) == 0 or np.sum(bg_mask) == 0:
        print("Error: No foreground or background scribbles found")
        return False
    
    # Set up video writer for original video with alpha
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
    
    # Create a temporary directory for frames if needed
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Process each frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
    frame_idx = 0
    
    print(f"Starting to process {frame_count} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_idx+1}/{frame_count}")
        
        # Process the frame to get alpha matte
        alpha_matte = process_frame(frame, fg_mask, bg_mask)
        
        # Create a green screen background
        green_bg = np.zeros_like(frame)
        green_bg[:,:] = (0, 255, 0)  # Green in BGR
        
        # Composite the foreground over the green background
        alpha_3channel = np.stack([alpha_matte] * 3, axis=2)
        composite = (frame * alpha_3channel + green_bg * (1 - alpha_3channel)).astype(np.uint8)
        
        # Write the composite frame
        out.write(composite)
        
        # Save the alpha matte for reference
        alpha_path = os.path.join(temp_dir, f"alpha_{frame_idx:04d}.png")
        cv2.imwrite(alpha_path, (alpha_matte * 255).astype(np.uint8))
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    print("Video processing complete!")
    return True
