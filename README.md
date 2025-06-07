# MultiVideoLab

Matting


Go to Code FolderFor matting, 

We used the https://github.com/reproductible-research/image-matting-with-a-closed-form-solution


cd image-matting-with-a-closed-form-solution
conda create -n matting python=3.9 -y
conda activate matting

pip install -r requirements.txt

first we Extract the human_clip_mp4

run: python extract_frame.py

AFter that we create scribble image template

and run the bath_matting.py

after that create the output_video.mp4