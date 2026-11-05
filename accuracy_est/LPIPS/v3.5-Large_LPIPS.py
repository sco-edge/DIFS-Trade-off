import sys
import torch
import json
import random
import csv
import os
import gc
import time
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline
import lpips
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

sys.stdout.reconfigure(line_buffering=True)

VERSION = "sd3.5-large"
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
H, W = 1024, 1024
BATCH_SIZE_QUALITY = 1   # CLIP, LPIPS용
BATCH_SIZE_LATENCY = 8   # Latency용
TOTAL_IMAGES_QUALITY = 1000  # CLIP, LPIPS
TOTAL_IMAGES_LATENCY = 300   # Latency
SEED = 42
REFERENCE_STEP = 100

step_sizes = [4, 6, 8, 10, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
ref_images_path      = f"{base_path}/ref_{VERSION}_T{REFERENCE_STEP}"
csv_output_file      = f"{base_path}/results/{VERSION}_metrics.csv"

lpips_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_coco_prompts(path, n):
    print("[INFO] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

print(f"[INFO] Loading {MODEL_ID}...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)
prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES_QUALITY)

print("[INFO] Loading LPIPS model (VGG)...")
loss_fn_lpips = lpips.LPIPS(net='vgg').cuda()
loss_fn_lpips.eval()

print("[INFO] Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Warm-up
print("[INFO] Warm-up...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=20, height=H, width=W)
torch.cuda.synchronize()
print("[INFO] Warm-up done!\n")

# T=100 기준 이미지 생성
if not os.path.exists(ref_images_path) or len(os.listdir(ref_images_path)) < TOTAL_IMAGES_QUALITY:
    print(f"[INFO] Generating reference images (T={REFERENCE_STEP})...")
    os.makedirs(ref_images_path, exist_ok=True)
    seed_everything(SEED)

    with torch.inference_mode():
        for i in range(0, TOTAL_IMAGES_QUALITY, BATCH_SIZE_QUALITY):
            batch_prompts = prompt_pool[i : i + BATCH_SIZE_QUALITY]
            if not batch_prompts:
                break
            generator = torch.Generator(device="cuda").manual_seed(SEED + i)
            output = pipe(
                prompt=batch_prompts,
                num_inference_steps=REFERENCE_STEP,
                height=H, width=W,
                generator=generator
            )
            for j, img in enumerate(output.images):
                img.save(os.path.join(ref_images_path, f"{i+j:05d}.png"))
            del output
            torch.cuda.empty_cache()

    print(f"[INFO] Reference images saved: {ref_images_path}\n")
else:
    print(f"[INFO] Reference images already exist: {ref_images_path}")

ref_fnames = sorted(os.listdir(ref_images_path))[:TOTAL_IMAGES_QUALITY]

os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
if not os.path.exists(csv_output_file):
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "LPIPS", "CLIP_Score", "User_Latency_sec", "NumImages", "Resolution"])

print(f"{'Steps':<8} | {'LPIPS':<10} | {'CLIP':<10} | {'UserLat(s)':<12}")
print("-" * 48)

for T in step_sizes:
    seed_everything(SEED)
    try:
        torch.cuda.empty_cache()
        gc.collect()

        total_lpips = 0.0
        total_clip  = 0.0
        count = 0

        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES_QUALITY, BATCH_SIZE_QUALITY):
                batch_prompts = prompt_pool[i : i + BATCH_SIZE_QUALITY]
                batch_fnames  = ref_fnames[i : i + BATCH_SIZE_QUALITY]
                if not batch_prompts:
                    break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                output = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                )

                ref_tensors = torch.stack([
                    lpips_transform(Image.open(os.path.join(ref_images_path, f)).convert('RGB'))
                    for f in batch_fnames
                ]).cuda()
                gen_tensors = torch.stack([
                    lpips_transform(img) for img in output.images
                ]).cuda()
                d = loss_fn_lpips(ref_tensors, gen_tensors)
                total_lpips += d.sum().item()

                clip_inputs = clip_processor(
                    text=batch_prompts,
                    images=output.images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to("cuda")
                clip_outputs = clip_model(**clip_inputs)
                total_clip += clip_outputs.logits_per_image.diag().sum().item()

                count += len(batch_prompts)
                del ref_tensors, gen_tensors, d, output, clip_inputs, clip_outputs
                torch.cuda.empty_cache()

        # User Latency
        torch.cuda.synchronize()
        u_start = time.time()
        with torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            _ = pipe(
                prompt=prompt_pool[:BATCH_SIZE_LATENCY],  # Latency: 배치 1번
                num_inference_steps=T,
                height=H, width=W,
                generator=generator
            )
        torch.cuda.synchronize()
        user_latency = time.time() - u_start

        lpips_score = total_lpips / TOTAL_IMAGES_QUALITY
        clip_score  = total_clip  / TOTAL_IMAGES_QUALITY

        print(f"{T:<8} | {lpips_score:<10.4f} | {clip_score:<10.4f} | {user_latency:<12.4f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, lpips_score, clip_score, user_latency, TOTAL_IMAGES_QUALITY, f"{H}x{W}"])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", "OOM", "OOM", TOTAL_IMAGES_QUALITY, f"{H}x{W}"])
        else:
            print(f"{T:<8} | ERROR: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", "ERROR", "ERROR", TOTAL_IMAGES_QUALITY, f"{H}x{W}"])
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")