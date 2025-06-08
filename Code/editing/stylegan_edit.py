import argparse, pathlib, torch, sys
from PIL import Image
from tqdm import tqdm
import numpy as np

# encoder4editing modülleri
sys.path.append("./encoder4editing")
from utils.model_utils import setup_model
from scripts.inference import get_latents

device = "cuda"

def pil_to_tensor(img):
    # e4e genelde 256x256 ve normalize edilmiş tensor ister
    img = img.resize((256, 256))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.tensor(arr).unsqueeze(0).to(device)  # [1,3,256,256]
    return tensor

def edit_one(image_path, net, direction, strength=2.0):
    img = Image.open(image_path).convert("RGB")
    img_tensor = pil_to_tensor(img)
    latent = get_latents(net, img_tensor)[0]
    edited_latent = latent + strength * direction
    out_img, _ = net.decoder([edited_latent.unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
    out_img = out_img[0].detach().cpu().numpy().transpose(1,2,0) * 255
    out_img = Image.fromarray(out_img.astype(np.uint8))
    return out_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dir", default="age")
    args = parser.parse_args()

    # Yön vektörü
    direction = torch.load(f"directions/{args.dir}.pt").to(device)

    # Encoder yükle
    net, opts = setup_model("encoder4editing/pretrained_models/e4e_ffhq_encode.pt", device)

    in_dir  = pathlib.Path(f"frames_raw/{args.tag}")
    out_dir = pathlib.Path(f"frames_stylegan/{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(sorted(in_dir.glob("*.png"))):
        edited = edit_one(frame, net, direction, strength=2.0)
        edited.save(out_dir / frame.name)
