import cv2
import os

video_path = "human_clip.mp4"
output_dir = "human_input_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_path, frame_resized)
    frame_idx += 1

cap.release()
print(f"Extracted {frame_idx} frames to {output_dir}/")
