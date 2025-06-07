import cv2
import numpy as np

img = cv2.imread('scribble_frame_0000.png', cv2.IMREAD_COLOR)
white_pixels = np.sum(np.all(img == [255, 255, 255], axis=2))
black_pixels = np.sum(np.all(img == [0, 0, 0], axis=2))
print("Tam beyaz piksel:", white_pixels)
print("Tam siyah piksel:", black_pixels)