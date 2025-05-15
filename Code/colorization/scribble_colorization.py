
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

def colorize_image(image_gray, marked_image, mask):
    """
    Colorize a grayscale image using the optimization-based colorization method.
    
    Parameters:
        image_gray: Grayscale image (single channel)
        marked_image: Color reference image with scribbles (BGR)
        mask: Binary mask indicating scribbled areas
    
    Returns:
        Colorized image in BGR format
    """
    # Convert to YUV color space
    marked_yuv = cv2.cvtColor(marked_image, cv2.COLOR_BGR2YUV)
    
    # Get dimensions
    height, width = image_gray.shape
    n_pixels = height * width
    
    # Create index matrix for easy reference
    indices = np.arange(n_pixels).reshape(height, width)
    
    # Create adjacency weights based on grayscale intensity differences
    weights = compute_weights(image_gray)
    
    # Set up the linear system for u channel
    print("Setting up sparse matrix for U channel...")
    A = construct_sparse_matrix(weights, indices, mask)
    bu = construct_right_side(marked_yuv[:,:,1], mask, indices)
    
    # Solve the system for u channel
    print("Solving for U channel...")
    u = spsolve(A, bu).reshape(height, width)
    
    # Set up the linear system for v channel
    print("Setting up sparse matrix for V channel...")
    bv = construct_right_side(marked_yuv[:,:,2], mask, indices)
    
    # Solve the system for v channel
    print("Solving for V channel...")
    v = spsolve(A, bv).reshape(height, width)
    
    # Create YUV result and convert to BGR
    print("Creating final colorized image...")
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[:,:,0] = image_gray
    result[:,:,1] = np.clip(u, 0, 255).astype(np.uint8)
    result[:,:,2] = np.clip(v, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(result, cv2.COLOR_YUV2BGR)

def compute_weights(gray_image, epsilon=1e-6):
    """
    Compute weights based on intensity differences between neighboring pixels.
    
    Parameters:
        gray_image: Grayscale image
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Tuple of (horizontal_weights, vertical_weights)
    """
    height, width = gray_image.shape
    gray_image = gray_image.astype(np.float64) / 255.0
    
    # Compute horizontal and vertical differences
    h_diff = np.abs(gray_image[:, 1:] - gray_image[:, :-1])
    v_diff = np.abs(gray_image[1:, :] - gray_image[:-1, :])
    
    # Compute weights using the formula from the paper
    h_weights = 1.0 / (h_diff + epsilon)
    v_weights = 1.0 / (v_diff + epsilon)
    
    return h_weights, v_weights

def construct_sparse_matrix(weights, indices, mask):
    """
    Construct the sparse matrix for the optimization problem.
    
    Parameters:
        weights: Tuple of (horizontal_weights, vertical_weights)
        indices: Matrix of indices for each pixel
        mask: Binary mask indicating scribbled areas
    
    Returns:
        Sparse matrix A for the linear system Ax = b
    """
    height, width = indices.shape
    n_pixels = height * width
    h_weights, v_weights = weights
    
    # Initialize arrays for sparse matrix construction
    row_indices = []
    col_indices = []
    data = []
    
    # Add diagonal elements for known (masked) pixels
    for i in range(height):
        for j in range(width):
            idx = indices[i, j]
            if mask[i, j] > 0:  # Known pixel
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(1.0)
            else:  # Unknown pixel
                # Add diagonal element
                row_indices.append(idx)
                col_indices.append(idx)
                
                # Calculate the sum of weights
                weight_sum = 0.0
                
                # Right neighbor
                if j < width - 1:
                    weight_sum += h_weights[i, j]
                
                # Left neighbor
                if j > 0:
                    weight_sum += h_weights[i, j-1]
                
                # Bottom neighbor
                if i < height - 1:
                    weight_sum += v_weights[i, j]
                
                # Top neighbor
                if i > 0:
                    weight_sum += v_weights[i-1, j]
                
                data.append(weight_sum)
                
                # Add off-diagonal elements
                # Right neighbor
                if j < width - 1:
                    row_indices.append(idx)
                    col_indices.append(indices[i, j+1])
                    data.append(-h_weights[i, j])
                
                # Left neighbor
                if j > 0:
                    row_indices.append(idx)
                    col_indices.append(indices[i, j-1])
                    data.append(-h_weights[i, j-1])
                
                # Bottom neighbor
                if i < height - 1:
                    row_indices.append(idx)
                    col_indices.append(indices[i+1, j])
                    data.append(-v_weights[i, j])
                
                # Top neighbor
                if i > 0:
                    row_indices.append(idx)
                    col_indices.append(indices[i-1, j])
                    data.append(-v_weights[i-1, j])
    
    # Create sparse matrix
    A = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_pixels, n_pixels))
    return A

def construct_right_side(channel, mask, indices):
    """
    Construct the right side of the linear system.
    
    Parameters:
        channel: Color channel (U or V)
        mask: Binary mask indicating scribbled areas
        indices: Matrix of indices for each pixel
    
    Returns:
        Vector b for the linear system Ax = b
    """
    height, width = indices.shape
    n_pixels = height * width
    b = np.zeros(n_pixels)
    
    for i in range(height):
        for j in range(width):
            idx = indices[i, j]
            if mask[i, j] > 0:  # Known pixel
                b[idx] = channel[i, j]
    
    return b

def process_video(input_path, output_path):
    """
    Process a video by colorizing each frame using scribble-based colorization.
    This method assumes you already have a marked image with scribbles.
    
    Parameters:
        input_path: Path to the input grayscale video
        output_path: Path to save the colorized video
        marked_image: Color reference image with scribbles (BGR)
        mask: Binary mask indicating scribbled areas
    """
    # This function is implemented in run_colorization.py as process_video_with_ui
    # It's included here as a placeholder and for documentation
    pass

def setup_colorization_ui(image, window_name="Draw Scribbles"):
    """
    Set up a UI for drawing color scribbles on an image.
    
    Parameters:
        image: Grayscale image to colorize
        window_name: Name of the window
    
    Returns:
        drawing_image: Color reference image with scribbles (BGR)
        mask: Binary mask indicating scribbled areas
    """
    # Create a copy for drawing scribbles
    drawing_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(image)
    
    # Current drawing color and state
    current_color = (0, 0, 255)  # Start with red (in BGR)
    brush_size = 5
    drawing = False
    last_point = None
    
    # Window setup
    cv2.namedWindow(window_name)
    
    # Function to handle mouse events
    def draw_scribble(event, x, y, flags, param):
        nonlocal drawing, last_point
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(drawing_image, last_point, (x, y), current_color, brush_size)
                cv2.line(mask, last_point, (x, y), 255, brush_size)
                last_point = (x, y)
                cv2.imshow(window_name, drawing_image)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    # Set the mouse callback
    cv2.setMouseCallback(window_name, draw_scribble)
    
    print("\nInstructions:")
    print("- Draw colored scribbles with the left mouse button")
    print("- Press 'r' for red, 'g' for green, 'b' for blue, 'y' for yellow, 'p' for purple")
    print("- Press '+' to increase brush size, '-' to decrease")
    print("- Press 'c' when done drawing to start colorizing")
    print("- Press 'q' to cancel\n")
    
    # Main interaction loop
    while True:
        cv2.imshow(window_name, drawing_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            current_color = (0, 0, 255)  # Red in BGR
            print("Selected color: Red")
        elif key == ord('g'):
            current_color = (0, 255, 0)  # Green in BGR
            print("Selected color: Green")
        elif key == ord('b'):
            current_color = (255, 0, 0)  # Blue in BGR
            print("Selected color: Blue")
        elif key == ord('y'):
            current_color = (0, 255, 255)  # Yellow in BGR
            print("Selected color: Yellow")
        elif key == ord('p'):
            current_color = (255, 0, 255)  # Purple in BGR
            print("Selected color: Purple")
        elif key == ord('+'):
            brush_size = min(30, brush_size + 1)
            print(f"Brush size: {brush_size}")
        elif key == ord('-'):
            brush_size = max(1, brush_size - 1)
            print(f"Brush size: {brush_size}")
        elif key == ord('c'):
            print("Finished drawing scribbles.")
            break
        elif key == ord('q'):
            print("Cancelled.")
            return None, None
    
    cv2.destroyWindow(window_name)
    return drawing_image, mask

def process_video(input_path, output_path):
    """
    Process a video by colorizing each frame using scribble-based colorization.
    
    Parameters:
        input_path: Path to the input grayscale video
        output_path: Path to save the colorized video
    
    Returns:
        True if successful, False otherwise
    """
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
    
    # Read the first frame for scribbling
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return False
    
    # Convert to grayscale (if not already)
    if len(first_frame.shape) == 3 and first_frame.shape[2] == 3:
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    else:
        first_frame_gray = first_frame.copy()
    
    # Use the UI to get scribbles
    drawing_image, mask = setup_colorization_ui(first_frame_gray, "Draw Scribbles for Video")
    if drawing_image is None or mask is None:
        return False  # User cancelled
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
    
    # Process each frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale (if not already)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Use the scribbles to colorize the current frame
        colorized = colorize_image(gray, drawing_image, mask)
        
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

def initialize_scribbles_automatically(gray_image, n_scribbles=10):
    """
    Automatically initialize some scribbles for demonstration purposes.
    This is useful for batch processing or testing.
    
    Parameters:
        gray_image: Grayscale image
        n_scribbles: Number of random scribbles to generate
    
    Returns:
        marked_image: Color reference image with scribbles (BGR)
        mask: Binary mask indicating scribbled areas
    """
    height, width = gray_image.shape
    marked_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(gray_image)
    
    # Define some colors (BGR)
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 0),    # Dark blue
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark red
        (128, 128, 0),  # Dark cyan
        (128, 0, 128),  # Dark magenta
        (0, 128, 128),  # Dark yellow
    ]
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_scribbles):
        # Random start and end points
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(max(0, x1-100), min(width, x1+100)), np.random.randint(max(0, y1-100), min(height, y1+100))
        
        # Random color from the list
        color = colors[i % len(colors)]
        
        # Draw the scribble line
        cv2.line(marked_image, (x1, y1), (x2, y2), color, 5)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 5)
    
    return marked_image, mask
