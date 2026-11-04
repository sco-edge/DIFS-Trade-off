"""Benchmark SDXL 1.0 latency, throughput, and GPU memory."""

import sys
import torch
import time
import json
import random
import csv
import os
import gc
import numpy as np

from diffusers import StableDiffusionXLPipeline

sys.stdout.reconfigure(line_buffering=True)

# Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Experiment Configuration
device = "cuda"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "SDXL_scaling.csv"

total_images = 300
batch_sizes  = [1, 2, 4, 8, 16, 32, 64, 80, 96, 128]
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
H, W         = 1024, 1024

# Data Loading
def load_coco_prompts(json_path, num_samples):
    print("[INFO] Loading COCO prompts...")
    if not os.path.exists(json_path):
        print(f"[WARNING] Path not found: {json_path}")
        return ["a photo of a cat"] * num_samples
    with open(json_path, "r") as f:
        data = json.load(f)
    captions = sorted(list(set([ann["caption"] for ann in data["annotations"]])))
    return captions[:num_samples]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# Model Loading (SDXL 1.0)
print("[INFO] Loading SDXL 1.0 (float32)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True
).to(device)

pipe.enable_attention_slicing()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    pass

pipe.set_progress_bar_config(disable=True)

# Warm-up
print("[INFO] Starting Warm-up...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=10, height=H, width=W).images
torch.cuda.synchronize()
print("[INFO] Warm-up completed! Starting benchmark.\n")

# CSV Initialization
with open(csv_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Batch", "Steps",
        "User_Latency_s",
        "Throughput_img_s",
        "Peak_Mem_GB"
    ])

print(f"{'Batch':<6} | {'Steps':<6} | {'UserLat':<9} | {'Throughput':<12} | {'PeakMem':<9}")
print("-" * 55)

# Main Evaluation Loop
for T in step_sizes:
    for B in batch_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # 1. User Latency & Throughput (Single batch response time)
            start_time = time.time()

            with torch.inference_mode():
                generator = torch.Generator(device="cuda").manual_seed(SEED)
                _ = pipe(
                    prompt_pool[:B],
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                ).images

            torch.cuda.synchronize()
            user_latency = time.time() - start_time
            
            throughput = B / user_latency
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3

            print(f"{B:<6} | {T:<6} | {user_latency:<9.4f} | {throughput:<12.2f} | {peak_mem:<9.2f}")

            with open(csv_output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([B, T, user_latency, throughput, peak_mem])

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{B:<6} | {T:<6} | OOM")
                with open(csv_output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([B, T, "OOM", "OOM", "OOM"])
                torch.cuda.empty_cache()
                gc.collect()
                break  # When OOM occurs in that Step size, skip the larger batch and move to the next Step
            else:
                raise e

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")
