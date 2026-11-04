"""Measure Stable Diffusion 3.5 text–image alignment with CLIP."""

import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np

from diffusers import StableDiffusion3Pipeline
from transformers import CLIPProcessor, CLIPModel

sys.stdout.reconfigure(line_buffering=True)

# Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Experiment Configuration
device = "cuda"
total_images = 1000
batch_size   = 10
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
H, W         = 1024, 1024

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/SD3.5_clip.csv"

# Data Loading
def load_coco_prompts(path, n):
    print("[INFO] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    captions = sorted(captions)
    return captions[:n]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# Model Loading (SD 3.5 Medium)
print("[INFO] Loading SD3.5 Medium (1024x1024)...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
).to(device)

pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)

# Model Loading (CLIP)
print("[INFO] Loading CLIP model...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# CLIP Score Calculation Function
def calc_clip_score(images, prompts):
    inputs = clip_processor(
        text=prompts, images=images,
        return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        outputs    = clip_model(**inputs)
        img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores = (img_embeds * txt_embeds).sum(dim=-1)
    return scores.cpu().float().tolist()

# Warm-up
print("[INFO] Starting Warm-up...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=20, height=H, width=W)
torch.cuda.synchronize()
print("[INFO] Warm-up completed! Starting benchmark.\n")

# CSV Initialization
if os.path.dirname(csv_output_file):
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
    
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Steps", "CLIP_Score"])

print(f"{'Steps':<8} | {'CLIP_Score':<12}")
print("-" * 23)

# Main Evaluation Loop
for T in step_sizes:
    try:
        torch.cuda.empty_cache()
        gc.collect()
        all_scores = []
        
        with torch.inference_mode():
            for i in range(0, total_images, batch_size):
                batch_prompts = prompt_pool[i:i+batch_size]
                if not batch_prompts: 
                    break
                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                images = pipe(batch_prompts, num_inference_steps=T,
                             height=H, width=W, generator=generator).images
                all_scores.extend(calc_clip_score(images, batch_prompts))
                
        clip_score = float(np.mean(all_scores))
        print(f"{T:<8} | {clip_score:<12.4f}")
        
        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, clip_score])
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM"])
        else:
            print(f"{T:<8} | ERROR")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR"])
        torch.cuda.empty_cache()
        gc.collect()
        
    finally:
        torch.cuda.empty_cache()

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")
