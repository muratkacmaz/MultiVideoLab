import os
import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import logging
from scipy.sparse import spdiags, csr_matrix
from tqdm import tqdm

def compute_matting_laplacian(image, constraints_map, epsilon=1e-5, window_radius=1):

    window_size = (window_radius * 2 + 1) ** 2
    height, width, channels = image.shape
    num_pixels = width * height
    pixel_indices = np.arange(num_pixels).reshape(height, width)
    pair_relations = int(((1 - constraints_map[window_radius:-window_radius, window_radius:-window_radius]).sum()) * (window_size ** 2))
    row_indices = np.zeros(pair_relations)
    col_indices = np.zeros(pair_relations)
    values = np.zeros(pair_relations)
    idx = 0
    for col in range(window_radius, width - window_radius):
        for row in range(window_radius, height - window_radius):
            if constraints_map[row, col]:
                continue
            window_indices = pixel_indices[row - window_radius: row + window_radius + 1, col - window_radius: col + window_radius + 1].flatten()
            window_pixels = image[row - window_radius: row + window_radius + 1, col - window_radius: col + window_radius + 1, :].reshape(window_size, channels)
            mean_window = np.mean(window_pixels, axis=0)
            covariance_inv = np.linalg.inv(
                (window_pixels.T @ window_pixels / window_size) - np.outer(mean_window, mean_window) + epsilon / window_size * np.eye(channels)
            )
            centered_window_pixels = window_pixels - mean_window
            local_affinity = (1 + centered_window_pixels @ covariance_inv @ centered_window_pixels.T) / window_size
            row_indices[idx:idx + window_size ** 2] = np.repeat(window_indices, window_size)
            col_indices[idx:idx + window_size ** 2] = np.tile(window_indices, window_size)
            values[idx:idx + window_size ** 2] = local_affinity.flatten()
            idx += window_size ** 2
    row_indices = row_indices[:idx]
    col_indices = col_indices[:idx]
    values = values[:idx]
    affinity_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(num_pixels, num_pixels))
    diagonal_matrix = spdiags(affinity_matrix.sum(axis=1).flatten(), 0, num_pixels, num_pixels)
    laplacian_matrix = diagonal_matrix - affinity_matrix
    return laplacian_matrix

def process_frame_dynamic(image_path, scribbles_path, output_path, epsilon=1e-5, radius=1):

    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR) / 255.0
    scribbles_img_raw = cv2.imread(scribbles_path, cv2.IMREAD_COLOR)
    scribbles_image = scribbles_img_raw / 255.0

    if original_image.shape != scribbles_image.shape:
        print(f"Error: Dimension mismatch for {image_path}")
        return False

    fg_mask = np.all(scribbles_img_raw == [255,255,255], axis=2)
    bg_mask = np.all(scribbles_img_raw == [0,0,0], axis=2)
    initial_alpha = np.zeros(fg_mask.shape, dtype=np.float32) + 0.5
    initial_alpha[fg_mask] = 1.0
    initial_alpha[bg_mask] = 0.0

    known_alpha_mask = 1 * (initial_alpha != 0.5)
    scribble_confidence = 100
    confidence_diagonal = scribble_confidence * scipy.sparse.diags(known_alpha_mask.flatten())
    foreground_alpha_mask = 1 * (initial_alpha == 1.0)

    laplacian_matrix = compute_matting_laplacian(original_image, known_alpha_mask, epsilon, radius)
    refined_alpha = scipy.sparse.linalg.spsolve(
        laplacian_matrix + confidence_diagonal,
        scribble_confidence * foreground_alpha_mask.flatten()
    )
    final_alpha = np.clip(refined_alpha.reshape(initial_alpha.shape), 0, 1)

    original_img = (original_image * 255).astype(np.uint8)
    alpha_3_channel = np.stack([final_alpha]*3, axis=2)
    foreground = original_img * alpha_3_channel
    foreground_bgr_alpha = cv2.cvtColor(foreground.astype(np.uint8), cv2.COLOR_BGR2BGRA)
    foreground_bgr_alpha[:, :, 3] = (final_alpha * 255).astype(np.uint8)
    cv2.imwrite(output_path, foreground_bgr_alpha)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch Closed-form Image Matting")
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with input frames')
    parser.add_argument('--scribble', type=str, required=True, help='Scribble image (tek bir tane)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--radius', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frame_list = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith('.png')])

    for fname in tqdm(frame_list):
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname.replace('.png', '_rgba.png'))
        process_frame_dynamic(input_path, args.scribble, output_path, epsilon=args.epsilon, radius=args.radius)
