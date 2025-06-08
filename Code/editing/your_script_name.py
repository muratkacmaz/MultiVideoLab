import cv2
import numpy as np
import os
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline # InstructPix2Pix
import sys
import gc
import warnings

warnings.filterwarnings("ignore")

class VideoFrameProcessor:
    def video_to_frames(self, video_path, output_dir, max_frames=50, target_size=(512, 512)):
        """
        Extracts frames from video, resizes them, and saves as PNG.
        target_size: (width, height) for resizing
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        os.makedirs(output_dir, exist_ok=True)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = min(max_frames, total_frames) 
        skip_frames = max(1, total_frames // max_frames)
        
        print(f"Extracting {max_frames} frames from {video_path} (skipping {skip_frames} frames per capture)...")

        while True:
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break
                
            if frame_count % skip_frames == 0:
                # Resize frame to target_size
                if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                
                frame_path = os.path.join(output_dir, f"frame_{len(frames):04d}.png")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {len(frames)} frames to {output_dir}")
        return frames

    def frames_to_video(self, frames_dir, output_path, fps=30):
        """
        Combines PNG frames back into a video.
        """
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.png')])
        
        if not frame_files:
            print(f"No PNG frames found in {frames_dir} to create video.")
            return
            
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            print(f"Error: Could not read first frame {first_frame_path}. Skipping video creation.")
            return

        height, width, layers = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for {output_path}. Check codec support.")
            return

        print(f"Creating video: {output_path} from {len(frame_files)} frames...")
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(os.path.join(frames_dir, frame_file))
            if frame is not None:
                out.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_file}. Skipping.")
            
            if (i + 1) % 10 == 0:
                print(f"  Added {i + 1}/{len(frame_files)} frames to video.")
                
        out.release()
        print(f"Video created: {output_path}")

    def clear_gpu_memory(self):
        """Clears GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

class ProjectVideoEditor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.processor = VideoFrameProcessor()
        self.instruct_pix2pix_pipeline = None

    def load_instruct_pix2pix(self):
        if self.device == "cuda":
            print("Loading InstructPix2Pix model...")
            try:

                self.instruct_pix2pix_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix",
                    torch_dtype=torch.float16, 
                    safety_checker=None
                )
                self.instruct_pix2pix_pipeline.to(self.device) # Modeli GPU'ya taşı
                self.instruct_pix2pix_pipeline.enable_attention_slicing()
                self.instruct_pix2pix_pipeline.enable_vae_slicing()
                # self.instruct_pix2pix_pipeline.enable_model_cpu_offload() # Daha az VRAM için
                print("InstructPix2Pix model loaded.")
            except Exception as e:
                print(f"Error loading InstructPix2Pix: {e}")
                print("InstructPix2Pix will not be available. Please ensure you have enough VRAM and a compatible CUDA setup.")
                self.instruct_pix2pix_pipeline = None
        else:
            print("InstructPix2Pix requires CUDA. Skipping model loading.")

    def apply_instruct_pix2pix_edit(self, input_frames_dir, output_frames_dir, instruction_prompt):
        if not self.instruct_pix2pix_pipeline:
            print("InstructPix2Pix model not loaded. Cannot apply edit.")
            return

        os.makedirs(output_frames_dir, exist_ok=True)
        frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.lower().endswith('.png')])
        print(f"Applying InstructPix2Pix edit: '{instruction_prompt}' to {len(frame_files)} frames...")

        for i, frame_file in enumerate(frame_files):
            self.processor.clear_gpu_memory() # Her karede belleği temizle
            try:
                image_path = os.path.join(input_frames_dir, frame_file)
                input_image = Image.open(image_path).convert("RGB")
                
                if input_image.size != (512, 512):
                    input_image = input_image.resize((512, 512), Image.LANCZOS)

                edited_image = self.instruct_pix2pix_pipeline(
                    prompt=instruction_prompt,
                    image=input_image,
                    num_inference_steps=50,     
                    guidance_scale=7.5,         
                    image_guidance_scale=1.5   
                ).images[0]

                output_path = os.path.join(output_frames_dir, f"edited_ip2p_{frame_file}")
                edited_image.save(output_path)
                
                if (i + 1) % 5 == 0: # Her 5 karede bir ilerlemeyi göster
                    print(f"Processed {i + 1}/{len(frame_files)} frames with InstructPix2Pix.")

            except Exception as e:
                print(f"Error processing frame {frame_file} with InstructPix2Pix: {e}")

                original_img = Image.open(image_path).convert("RGB")
                original_img.save(os.path.join(output_frames_dir, f"edited_ip2p_{frame_file}"))
            finally:
                self.processor.clear_gpu_memory()

        print(f"InstructPix2Pix processing complete. Output: {output_frames_dir}")


    def apply_image2stylegan_conceptual_edit(self, input_frames_dir, output_frames_dir, edit_type):
        os.makedirs(output_frames_dir, exist_ok=True)
        frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.lower().endswith('.png')])
        print(f"Applying Image2StyleGAN (Conceptual) edit: '{edit_type}' to {len(frame_files)} frames...")

        for i, frame_file in enumerate(frame_files):
            try:
                image_path = os.path.join(input_frames_dir, frame_file)
                input_image_pil = Image.open(image_path).convert("RGB")
                
                if input_image_pil.size != (512, 512):
                    input_image_pil = input_image_pil.resize((512, 512), Image.LANCZOS)
                
                img_cv2 = cv2.cvtColor(np.array(input_image_pil), cv2.COLOR_RGB2BGR)
                edited_img_cv2 = img_cv2.copy() # Başlangıçta kopyası


                if edit_type == "smile":
                    
                    edited_img_cv2 = cv2.convertScaleAbs(edited_img_cv2, alpha=1.1, beta=20)
                    hsv = cv2.cvtColor(edited_img_cv2, cv2.COLOR_BGR2HSV)
                    hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2) # Doygunluk
                    edited_img_cv2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                elif edit_type == "glasses":
                    
                    h, w, _ = edited_img_cv2.shape
                    glasses_color = (0, 0, 0) # Siyah
                    thickness = 2
                    
                    cv2.rectangle(edited_img_cv2, (int(w*0.35), int(h*0.4)), (int(w*0.48), int(h*0.5)), glasses_color, thickness)
                    
                    cv2.rectangle(edited_img_cv2, (int(w*0.52), int(h*0.4)), (int(w*0.65), int(h*0.5)), glasses_color, thickness)
                    
                    cv2.line(edited_img_cv2, (int(w*0.48), int(h*0.45)), (int(w*0.52), int(h*0.45)), glasses_color, thickness)
                
                edited_image_pil = Image.fromarray(cv2.cvtColor(edited_img_cv2, cv2.COLOR_BGR2RGB))
                
                output_path = os.path.join(output_frames_dir, f"edited_isg_{frame_file}")
                edited_image_pil.save(output_path)
                
                if (i + 1) % 5 == 0:
                    print(f"Processed {i + 1}/{len(frame_files)} frames with Image2StyleGAN (conceptual).")

            except Exception as e:
                print(f"Error processing frame {frame_file} with Image2StyleGAN (conceptual): {e}")
                # Hata durumunda orijinali kopyala
                original_img = Image.open(image_path).convert("RGB")
                original_img.save(os.path.join(output_frames_dir, f"edited_isg_{frame_file}"))
            finally:
                self.processor.clear_gpu_memory()

        print(f"Image2StyleGAN (conceptual) processing complete. Output: {output_frames_dir}")


def main():
    editor = ProjectVideoEditor()
    video_processor = VideoFrameProcessor()

    video_path = "videos/human_clip.mp4" 
    

    raw_frames_dir = os.path.join("frames_raw", "raw_frames") 
    os.makedirs(raw_frames_dir, exist_ok=True)


    if not os.path.exists(raw_frames_dir) or not os.listdir(raw_frames_dir):
        print("\n--- Step 1: Extracting Frames (and resizing to 512x512 for models) ---")
        
        video_processor.video_to_frames(video_path, raw_frames_dir, max_frames=50, target_size=(512, 512)) 
    else:
        print(f"\n--- Step 1: Raw frames already exist in {raw_frames_dir}. Skipping extraction. ---")

    if not os.listdir(raw_frames_dir):
        print("Error: No frames found in raw_frames_dir. Please check the path or run extraction.")
        sys.exit(1)



    print("\n" + "="*50)
    print("--- Applying Method A: InstructPix2Pix (Diffusion Model) ---")
    print("="*50)
    
    editor.load_instruct_pix2pix()
    if editor.instruct_pix2pix_pipeline:

        ip2p_smile_output_dir = os.path.join("project_final_outputs", "instructpix2pix_smile_frames")
        editor.apply_instruct_pix2pix_edit(raw_frames_dir, ip2p_smile_output_dir, 
                                            instruction_prompt="make her smile brightly")
        video_processor.frames_to_video(ip2p_smile_output_dir, os.path.join("project_final_outputs", "video_ip2p_smile.mp4"))


        ip2p_glasses_output_dir = os.path.join("project_final_outputs", "instructpix2pix_glasses_frames")
        editor.apply_instruct_pix2pix_edit(raw_frames_dir, ip2p_glasses_output_dir, 
                                            instruction_prompt="add stylish glasses to her, professional photo")
        video_processor.frames_to_video(ip2p_glasses_output_dir, os.path.join("project_final_outputs", "video_ip2p_glasses.mp4"))
    else:
        print("InstructPix2Pix was not loaded or failed. Skipping edits for this method.")


    print("\n" + "="*50)
    print("--- Applying Method B: Image2StyleGAN (GAN/Flow-Based Model - Conceptual) ---")
    print("="*50)
    print("NOTE: This section provides a conceptual demonstration using OpenCV filters.")
    print("      Actual Image2StyleGAN integration requires complex setup of external repositories.")
    

    isg_smile_output_dir = os.path.join("project_final_outputs", "image2stylegan_smile_frames_conceptual")
    editor.apply_image2stylegan_conceptual_edit(raw_frames_dir, isg_smile_output_dir, "smile")
    video_processor.frames_to_video(isg_smile_output_dir, os.path.join("project_final_outputs", "video_isg_smile_conceptual.mp4"))
    
    isg_glasses_output_dir = os.path.join("project_final_outputs", "image2stylegan_glasses_frames_conceptual")
    editor.apply_image2stylegan_conceptual_edit(raw_frames_dir, isg_glasses_output_dir, "glasses")
    video_processor.frames_to_video(isg_glasses_output_dir, os.path.join("project_final_outputs", "video_isg_glasses_conceptual.mp4"))
    
    print("\n--- All processing attempts complete ---")
    print(f"Check output videos and frame folders in: project_final_outputs/")


if __name__ == "__main__":
    main()