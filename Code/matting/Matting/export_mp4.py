import cv2
import os
from natsort import natsorted

input_dir = "output_rgba"
output_video = "output_video.mp4"
fps = 25 


files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
files = natsorted(files)


first_frame = cv2.imread(os.path.join(input_dir, files[0]), cv2.IMREAD_UNCHANGED)
height, width = first_frame.shape[:2]


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for fname in files:
    frame = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_UNCHANGED)
   
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    out.write(frame)

out.release()
print(f"{output_video} saved.")
