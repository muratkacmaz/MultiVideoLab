# scripts/diffusion_edit.py
import argparse, glob, pathlib, torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
from tqdm import tqdm

device = "cuda"

def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)
    return pipe

def center_crop_resize(im, size=512):
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im_cropped = im.crop((left, top, left+side, top+side)).resize((size, size))
    return im_cropped

def canny(image):
    import cv2, numpy as np
    img = cv2.imread(str(image))
    edges = cv2.Canny(img,50,150)
    return Image.fromarray(edges)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)      # face / street / dog
    parser.add_argument("--prompt", required=True)   # "add pink hair", ...
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pipe = load_pipeline()
    generator = torch.Generator(device).manual_seed(args.seed)

    in_dir  = pathlib.Path(f"frames_raw/{args.tag}")
    out_dir = pathlib.Path(f"frames_sd/{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(sorted(in_dir.glob("*.png"))):
        img  = Image.open(frame).convert("RGB")
        img  = center_crop_resize(img, 512)

        cimg = canny(frame)
        cimg = center_crop_resize(cimg, 512)

        out  = pipe(
            prompt=args.prompt,
            image=img,
            control_image=cimg,
            generator=generator,
            num_inference_steps=30,
            guidance_scale=4.0,
        ).images[0]
        out.save(out_dir / frame.name)
