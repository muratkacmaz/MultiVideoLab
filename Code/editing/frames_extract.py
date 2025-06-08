# scripts/frames_extract.py
import subprocess, pathlib, argparse

def extract(video_path: str, out_dir: str, fps=30):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "2",
        f"{out_dir}/%04d.png"
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    vids = {
        "animal":  "videos/animal_clip.mp4",
        "human":"videos/human_clip.mp4",
        "train":   "videos/train_clip.mp4"
    }
    for tag, path in vids.items():
        extract(path, f"frames_raw/{tag}")
