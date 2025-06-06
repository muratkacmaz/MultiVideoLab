#!/usr/bin/env python 
'''
Copyright (C) 2024 Aissa Abdelaziz, Mahdi Ranjbar, and Mohammad Ali Jauhar.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
'''

import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import logging
import sys
import argparse
from scipy.sparse import spdiags, csr_matrix


def compute_matting_laplacian(image, constraints_map, epsilon=1e-5, window_radius=1):
    '''
    Computes the Matting Laplacian matrix using the closed-form matting method.
    
    Args:
        image : Input image normalized to [0,1].
        constraints_map : Binary mask indicating known alpha values.
        epsilon : Regularization parameter to ensure numerical stability.
        window_radius : Radius of the local window.
    
    Returns:
        The Matting Laplacian sparse matrix.
    '''
    logging.info('Computing Matting Laplacian...')
    
    window_size = (window_radius * 2 + 1) ** 2  # Total pixels in a window
    height, width, channels = image.shape
    num_pixels = width * height
    pixel_indices = np.arange(num_pixels).reshape(height, width)
    
    # Estimate the number of nonzero elements for sparse matrix allocation
    pair_relations = int(((1 - constraints_map[window_radius:-window_radius, window_radius:-window_radius]).sum()) * (window_size ** 2))
    row_indices = np.zeros(pair_relations)
    col_indices = np.zeros(pair_relations)
    values = np.zeros(pair_relations)
    idx = 0
    
    for col in range(window_radius, width - window_radius):
        for row in range(window_radius, height - window_radius):
            if constraints_map[row, col]:
                continue  # Skip known pixels
            
            # Extract local window indices
            window_indices = pixel_indices[row - window_radius: row + window_radius + 1, col - window_radius: col + window_radius + 1].flatten()
            window_pixels = image[row - window_radius: row + window_radius + 1, col - window_radius: col + window_radius + 1, :].reshape(window_size, channels)
            # Eq. 23 in the paper
            # Compute mean and covariance of local window
            mean_window = np.mean(window_pixels, axis=0)
            covariance_inv = np.linalg.inv(
                (window_pixels.T @ window_pixels / window_size) - np.outer(mean_window, mean_window) + epsilon / window_size * np.eye(channels)
            )  
            # Compute affinity matrix within the window
            centered_window_pixels = window_pixels - mean_window
            local_affinity = (1 + centered_window_pixels @ covariance_inv @ centered_window_pixels.T) / window_size
            
            # Fill sparse matrix
            row_indices[idx:idx + window_size ** 2] = np.repeat(window_indices, window_size)
            col_indices[idx:idx + window_size ** 2] = np.tile(window_indices, window_size)
            values[idx:idx + window_size ** 2] = local_affinity.flatten()
            idx += window_size ** 2
    
    # Trim arrays to actual size
    row_indices = row_indices[:idx]
    col_indices = col_indices[:idx]
    values = values[:idx]
    
    # Construct sparse affinity matrix
    affinity_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(num_pixels, num_pixels))
    diagonal_matrix = spdiags(affinity_matrix.sum(axis=1).flatten(), 0, num_pixels, num_pixels)
    laplacian_matrix = diagonal_matrix - affinity_matrix
    
    return laplacian_matrix


def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Closed-form Image Matting")
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('-s', '--scribbles', type=str, required=True, help='Path to scribbles image')
    parser.add_argument('-e', '--epsilon', type=float, default=1e-5, choices=[1e-5, 1e-4, 1e-3, 1e-2], help='Regularization parameter epsilon')
    parser.add_argument('-r', '--radius', type=int, default=1, choices=[1, 2, 3, 4], help='Window radius for local matting')
    args = parser.parse_args()


    img1 = cv2.imread(args.image)
    img2 = cv2.imread(args.scribbles)

    print("Frame shape:", img1.shape)
    print("Scribble shape:", img2.shape)
    
    original_image = cv2.imread(args.image, cv2.IMREAD_COLOR) / 255.0
    scribbles_image = cv2.imread(args.scribbles, cv2.IMREAD_COLOR) / 255.0
    
    if original_image.shape != scribbles_image.shape:
        print("Error: Input image and scribbles must have the same dimensions.")
        sys.exit(1)
    
    # Changed from I - S to S - I as in the paper we say 1 is foreground. When doing the following 
    # operation as I - S, we get 1 as the background
    initial_alpha = np.sign(np.sum(scribbles_image - original_image, axis=2)) / 2 + 0.5
    known_alpha_mask = 1 * (initial_alpha != 0.5)
    
    scribble_confidence = 100
    # -- The following stuff has been commented out and replaced with alterative
    # -- expressions to match the equations in the paper
    # confidence_weights = scribble_confidence * known_alpha_mask
    # confidence_diagonal = scipy.sparse.diags(confidence_weights.flatten())
    confidence_diagonal = scribble_confidence * scipy.sparse.diags(known_alpha_mask.flatten())
    foreground_alpha_mask = 1 * (initial_alpha == 1.0)
    laplacian_matrix = compute_matting_laplacian(original_image, known_alpha_mask, args.epsilon, args.radius)
    
    logging.info('Solving the linear system for alpha matte...')
    refined_alpha = scipy.sparse.linalg.spsolve(
        laplacian_matrix + confidence_diagonal,
        scribble_confidence * foreground_alpha_mask.flatten()
    )
    # initial_alpha.flatten() * confidence_weights.flatten()
    
    final_alpha = np.clip(refined_alpha.reshape(initial_alpha.shape), 0, 1)
    
    # changed from 1 - final_alpha to final_alpha
    cv2.imwrite("output.png", (final_alpha) * 255.0)

    logging.info('Alpha matte saved as output.png')


    original_img_path = args.image # Sizin orijinal girdi görüntünüz
    alpha_matte_path = 'output.png'       # main.py'nin kaydettiği alpha matte

    original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    alpha_matte = cv2.imread(alpha_matte_path, cv2.IMREAD_GRAYSCALE) / 255.0 # Alpha'yı 0-1 arasına normalleştir

    # Orijinal resmin boyutlarıyla alfa matrisinin aynı olduğundan emin olun
    if original_img.shape[:2] != alpha_matte.shape:
        print("Boyut uyuşmazlığı! Orijinal resim ve alfa matrisi aynı boyutta olmalı.")
    else:
        # Alpha kanalını 3 kanallı hale getirerek orijinal resme uygulayın
        alpha_3_channel = cv2.merge([alpha_matte, alpha_matte, alpha_matte]) # Alpha'yı BGR kanallarına genişlet

        # Ön planı elde etmek için orijinal resmi alfa ile çarpın
        foreground = original_img * alpha_3_channel

        # Eğer ön planı şeffaf arka planla kaydetmek istiyorsanız (PNG gibi formatlar için)
        # 4 kanallı bir görüntü oluşturun (BGR + Alpha)
        foreground_bgr_alpha = cv2.cvtColor(foreground.astype(np.uint8), cv2.COLOR_BGR2BGRA)
        foreground_bgr_alpha[:, :, 3] = (alpha_matte * 255).astype(np.uint8) # Alfa kanalını doğrudan ayarla

        # Ön plan görüntüsünü kaydedin
        cv2.imwrite("foreground_extracted.png", foreground_bgr_alpha)
        print("Ön plan başarıyla çıkarıldı ve 'foreground_extracted.png' olarak kaydedildi.")



if __name__ == "__main__":
    main()