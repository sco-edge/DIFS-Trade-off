# DIFS

DIFS is an experiment repository for measuring how diffusion-model inference
settings affect latency, throughput, GPU memory use, and image quality. The
current scripts benchmark Stable Diffusion 1.5, Stable Diffusion 2.1, SDXL,
and Stable Diffusion 3.5 on a CUDA GPU.

## What is measured

| Area | Measurements |
| --- | --- |
| Performance | batch latency, throughput, peak GPU memory |
| Image quality | CLIP score, FID, LPIPS |
| Serving | request latency, SLO attainment, model-switch overhead, GPU utilization |

Most experiments sweep the denoising-step count so that their CSV outputs can
be compared as quality–latency trade-off curves.

## Repository layout

```text
.
├── accuracy_est/              # CLIP, FID, and LPIPS experiments
├── latency_est/               # Per-model latency and throughput sweeps
├── PaniniServe_scheduler.py   # Windowed scheduler and real-GPU replay
└── docs/                      # Setup and experiment documentation
```

See [docs/project-structure.md](docs/project-structure.md) for the complete
file map.

## Quick start

1. Create a Python environment and install the CUDA build of PyTorch.
2. Install the remaining packages from `requirements.txt`.
3. Download COCO 2014 validation captions and update
   `coco_annotation_path` in the script you plan to run.
4. Reduce image counts, batch sizes, and step sizes for a smoke test.
5. Run one experiment, for example:

   ```bash
   python latency_est/SD_v1.5.py
   ```

The scripts download model weights from Hugging Face on first use. Some model
repositories may require accepting their license and authenticating with the
Hugging Face CLI.

## Configuration note

The experiment scripts currently contain machine-specific absolute paths.
Review every path and output location before running a long job. A full FID
run generates thousands of images and may require substantial GPU time and
disk space.

## Current scope

The repository contains executable research scripts rather than a packaged
Python library. Results depend on the GPU, driver, CUDA and library versions,
model revision, and experiment configuration. Record those details alongside
each result before comparing runs from different machines.
