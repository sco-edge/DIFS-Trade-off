"""Measure Stable Diffusion 2.1 perceptual convergence with LPIPS."""

import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import lpips
import torchvision.transforms as transforms

sys.stdout.reconfigure(line_buffering=True)

# Experiment Configuration
VERSION = "v2.1"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
H, W = 768, 768
FIXED_BATCH_SIZE = 1
TOTAL_IMAGES = 1000
SEED = 42
REFERENCE_STEP = 100

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
ref_images_path      = f"{base_path}/ref_{VERSION}_T{REFERENCE_STEP}"
csv_output_file      = f"{base_path}/results/{VERSION}_LPIPS.csv"

# LPIPS Image Preprocessing
lpips_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Helper Functions
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_coco_prompts(path, n):
    print("[INFO] Loading COCO prompts...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

# Model Loading (SD v2.1)
print(f"[INFO] Loading {MODEL_ID}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    pass

pipe.set_progress_bar_config(disable=True)
prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# Model Loading (LPIPS)
print("[INFO] Loading LPIPS model (VGG)...")
loss_fn_lpips = lpips.LPIPS(net='vgg').cuda()
loss_fn_lpips.eval()

# Reference Image Generation (100 Steps)
if not os.path.exists(ref_images_path) or len(os.listdir(ref_images_path)) < TOTAL_IMAGES:
    print(f"[INFO] Generating reference images (T={REFERENCE_STEP})...")
    os.makedirs(ref_images_path, exist_ok=True)
    seed_everything(SEED)

    with torch.inference_mode():
        for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
            batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
            if not batch_prompts:
                break
            generator = torch.Generator(device="cuda").manual_seed(SEED + i)
            output = pipe(
                batch_prompts,
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

ref_fnames = sorted(os.listdir(ref_images_path))[:TOTAL_IMAGES]

# CSV Initialization
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
if not os.path.exists(csv_output_file):
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "LPIPS", "NumImages", "Resolution"])

print(f"{'Steps':<8} | {'LPIPS':<12}")
print("-" * 23)

# Main Evaluation Loop
for T in step_sizes:
    seed_everything(SEED)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        total_score = 0.0
        count = 0

        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
                batch_fnames  = ref_fnames[i : i + FIXED_BATCH_SIZE]
                if not batch_prompts:
                    break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                output = pipe(
                    batch_prompts,
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
                total_score += d.sum().item()
                count += len(batch_prompts)

                del ref_tensors, gen_tensors, d, output
                torch.cuda.empty_cache()

        lpips_score = total_score / count
        print(f"{T:<8} | {lpips_score:<12.4f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, lpips_score, TOTAL_IMAGES, f"{H}x{W}"])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", TOTAL_IMAGES, f"{H}x{W}"])
        else:
            print(f"{T:<8} | ERROR: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", TOTAL_IMAGES, f"{H}x{W}"])
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")
