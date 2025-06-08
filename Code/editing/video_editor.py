import cv2
import numpy as np
from PIL import Image
import torch
import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings("ignore")

# Check available resources
try:
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Diffusers not available. Using OpenCV-based methods.")
    DIFFUSERS_AVAILABLE = False

class MemoryEfficientVideoEditor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.setup_models()
        
    def setup_models(self):
        """Initialize lightweight models with memory optimization"""
        self.model_loaded = False
        self.pipe = None
        
        if DIFFUSERS_AVAILABLE and torch.cuda.is_available():
            try:
                # Use smaller, more memory efficient model
                print("Loading memory-efficient Stable Diffusion model...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,  # Half precision
                    variant="fp16",
                    use_safetensors=True
                )
                
                # Memory optimizations
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
                self.pipe.enable_model_cpu_offload()  # Move models to CPU when not in use
                
                print("Model loaded with memory optimizations")
                self.model_loaded = True
                
            except Exception as e:
                print(f"Error loading diffusion model: {e}")
                print("Using OpenCV-based methods instead")
                self.model_loaded = False
        else:
            print("Using OpenCV-based methods")
            
    def clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def video_to_frames(self, video_path, output_dir, max_frames=150):
        """Extract frames from video with limit"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Skip frames if video is too long
        skip_frames = max(1, total_frames // max_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret or len(frames) >= max_frames:
                break
                
            if frame_count % skip_frames == 0:
                # Resize frame to reduce memory usage
                height, width = frame.shape[:2]
                if width > 512:
                    scale = 512 / width
                    new_width = 512
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                frame_path = os.path.join(output_dir, f"frame_{len(frames):04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def frames_to_video(self, frames_dir, output_path, fps=30):
        """Combine frames back to video"""
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        if not frame_files:
            return
            
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width, layers = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(frames_dir, frame_file))
            out.write(frame)
            
        out.release()
        print(f"Created video: {output_path}")

    def method1_artistic_stylization(self, input_frames, output_dir):
        """Method 1: OpenCV artistic stylization"""
        os.makedirs(output_dir, exist_ok=True)
        print("Applying Method 1: Artistic Stylization")
        
        for i, frame_path in enumerate(input_frames):
            # Load frame
            image = cv2.imread(frame_path)
            
            # Apply artistic stylization
            styled_image = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
            
            # Save edited frame
            output_path = os.path.join(output_dir, f"method1_frame_{i:04d}.jpg")
            cv2.imwrite(output_path, styled_image)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(input_frames)} frames")
                
        return output_dir

    def method2_edge_preserving(self, input_frames, output_dir):
        """Method 2: Edge-preserving filter with color enhancement"""
        os.makedirs(output_dir, exist_ok=True)
        print("Applying Method 2: Edge-Preserving Filter")
        
        for i, frame_path in enumerate(input_frames):
            # Load frame
            image = cv2.imread(frame_path)
            
            # Apply edge-preserving filter
            filtered_image = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=0.4)
            
            # Enhance colors
            hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.2)  # Increase saturation
            enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Save edited frame
            output_path = os.path.join(output_dir, f"method2_frame_{i:04d}.jpg")
            cv2.imwrite(output_path, enhanced_image)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(input_frames)} frames")
                
        return output_dir

    def method3_diffusion_lightweight(self, input_frames, output_dir, batch_size=1):
        """Method 3: Lightweight diffusion editing (if available)"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.model_loaded:
            print("Diffusion model not available, using alternative...")
            return self.method_alternative_filters(input_frames, output_dir)
        
        print("Applying Method 3: Lightweight Diffusion")
        
        for i in range(0, len(input_frames), batch_size):
            batch_frames = input_frames[i:i+batch_size]
            
            for j, frame_path in enumerate(batch_frames):
                try:
                    # Clear memory before each frame
                    self.clear_gpu_memory()
                    
                    # Load and resize image
                    image = Image.open(frame_path)
                    image = image.resize((512, 512))  # Fixed size for memory efficiency
                    
                    # Simple prompt for style transfer
                    prompt = "artistic digital painting style, vibrant colors"
                    
                    # Generate with minimal steps and lower guidance
                    with torch.no_grad():
                        result = self.pipe(
                            prompt=prompt,
                            image=image,
                            num_inference_steps=10,  # Reduced steps
                            guidance_scale=5.0,      # Lower guidance
                            strength=0.3             # Lower strength
                        ).images[0]
                    
                    # Save result
                    output_path = os.path.join(output_dir, f"method3_frame_{i+j:04d}.jpg")
                    result.save(output_path)
                    
                    if (i + j + 1) % 5 == 0:
                        print(f"Processed {i + j + 1}/{len(input_frames)} frames")
                        
                except Exception as e:
                    print(f"Error processing frame {i+j}: {e}")
                    # Fallback: copy original
                    image = Image.open(frame_path)
                    output_path = os.path.join(output_dir, f"method3_frame_{i+j:04d}.jpg")
                    image.save(output_path)
                    
                finally:
                    self.clear_gpu_memory()
                    
        return output_dir

    def method_alternative_filters(self, input_frames, output_dir):
        """Alternative method using advanced OpenCV filters"""
        os.makedirs(output_dir, exist_ok=True)
        print("Applying Alternative Method: Advanced Filters")
        
        for i, frame_path in enumerate(input_frames):
            # Load frame
            image = cv2.imread(frame_path)
            
            # Apply pencil sketch effect
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            
            # Create colored version
            color = cv2.bilateralFilter(image, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            
            # Save edited frame
            output_path = os.path.join(output_dir, f"alt_method_frame_{i:04d}.jpg")
            cv2.imwrite(output_path, cartoon)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(input_frames)} frames")
                
        return output_dir

    def process_video(self, video_path, output_base_dir, video_name):
        """Complete video processing pipeline"""
        print(f"\n=== Processing {video_name} ===")
        
        # Create directories
        frames_dir = os.path.join(output_base_dir, video_name, "original_frames")
        method1_dir = os.path.join(output_base_dir, video_name, "method1_artistic")
        method2_dir = os.path.join(output_base_dir, video_name, "method2_edge_preserving")
        
        # Step 1: Extract frames
        print("Step 1: Extracting frames...")
        frames = self.video_to_frames(video_path, frames_dir, max_frames=100)  # Limit frames
        
        if not frames:
            print(f"No frames extracted from {video_path}")
            return None, None
        
        # Step 2: Apply Method 1
        print("Step 2: Applying Method 1...")
        self.method1_artistic_stylization(frames, method1_dir)
        
        # Step 3: Apply Method 2
        print("Step 3: Applying Method 2...")
        self.method2_edge_preserving(frames, method2_dir)
        
        # Step 4: Create output videos
        print("Step 4: Creating output videos...")
        method1_video = os.path.join(output_base_dir, video_name, f"{video_name}_method1.mp4")
        method2_video = os.path.join(output_base_dir, video_name, f"{video_name}_method2.mp4")
        
        self.frames_to_video(method1_dir, method1_video)
        self.frames_to_video(method2_dir, method2_video)
        
        print(f"✅ Completed processing {video_name}")
        return method1_video, method2_video

def main():
    # Initialize video editor
    editor = MemoryEfficientVideoEditor()
    
    # Define your 3 videos (update paths as needed)
    videos = [
        {"path": "videos/human_clip.mp4", "name": "human_video"}
    ]
    
    output_base_dir = "project_outputs"
    results = {}
    
    # Process each video
    for video_info in videos:
        video_path = video_info["path"]
        video_name = video_info["name"]
        
        if os.path.exists(video_path):
            try:
                method1_output, method2_output = editor.process_video(
                    video_path, output_base_dir, video_name
                )
                results[video_name] = {
                    "method1": method1_output,
                    "method2": method2_output
                }
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
        else:
            print(f"❌ Video not found: {video_path}")
    
    # Print results summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    for video_name, outputs in results.items():
        print(f"\n{video_name.upper()}:")
        print(f"  Method 1 (Artistic): {outputs['method1']}")
        print(f"  Method 2 (Edge-Preserving): {outputs['method2']}")
    
    print(f"\nAll outputs saved in: {output_base_dir}/")

if __name__ == "__main__":
    main()