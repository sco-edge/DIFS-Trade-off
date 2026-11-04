"""Measure Stable Diffusion 3.5 distribution quality with FID."""

import sys
import torch
import json
import random
import csv
import os
import gc
import shutil
import numpy as np
from diffusers import StableDiffusion3Pipeline
import torch_fidelity

sys.stdout.reconfigure(line_buffering=True)

# Experiment Configuration
VERSION = "v3.5"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
H, W = 1024, 1024
FIXED_BATCH_SIZE = 4  
TOTAL_IMAGES = 10000
SEED = 42

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"
generated_root       = f"{base_path}/generated_{VERSION}"
csv_output_file      = f"{base_path}/results/{VERSION}_FID.csv"

# Helper Functions
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

prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# Model Loading (SD 3.5 Medium)
print(f"[INFO] Loading {MODEL_ID}...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)

# CSV Initialization
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
if not os.path.exists(csv_output_file):
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "FID", "BatchSize", "Resolution"])

print(f"{'Steps':<8} | {'FID':<12}")
print("-" * 23)

# Main Evaluation Loop
for T in step_sizes:
    seed_everything(SEED)
    save_dir = os.path.join(generated_root, f"T{T}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
                if not batch_prompts: 
                    break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                output = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                )

                for j, img in enumerate(output.images):
                    img.save(os.path.join(save_dir, f"{i+j:05d}.png"))

        metrics = torch_fidelity.calculate_metrics(
            input1=real_images_path,
            input2=os.path.abspath(save_dir),
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True
        )
        fid = metrics['frechet_inception_distance']

        print(f"{T:<8} | {fid:<12.2f}")
        
        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, fid, FIXED_BATCH_SIZE, f"{H}x{W}"])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", FIXED_BATCH_SIZE, f"{H}x{W}"])
        else:
            print(f"{T:<8} | ERROR")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", FIXED_BATCH_SIZE, f"{H}x{W}"])
        torch.cuda.empty_cache()
        gc.collect()
    
    finally:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")
