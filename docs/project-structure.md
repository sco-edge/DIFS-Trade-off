# Repository structure

## Top level

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and quick start |
| `requirements.txt` | Shared Python dependencies |
| `PaniniServe_scheduler.py` | Scheduler simulation followed by real-GPU execution |
| `latency_est/` | Latency, throughput, and peak-memory sweeps |
| `accuracy_est/` | CLIP, FID, and LPIPS quality sweeps |
| `scripts/` | Lightweight utilities that do not run model inference |
| `docs/` | Reproduction and interpretation guides |

## Performance scripts

`latency_est/` contains one script for each checked-in model family:

- `SD_v1.5.py` — Stable Diffusion 1.5 at 512×512.
- `SD_v2.1.py` — Stable Diffusion 2.1 at 768×768.
- `SDXL.py` — SDXL 1.0 at 1024×1024.
- `SD_v3.5.py` — Stable Diffusion 3.5 Medium at 1024×1024.

Each script sweeps batch size and denoising steps, then writes a CSV with
latency, throughput, and peak allocated GPU memory.

## Quality scripts

`accuracy_est/` is split by metric:

- `CLIP/` measures text–image alignment.
- `FID/` compares distributions of generated and real images.
- `LPIPS/` compares generated images with high-step reference generations.

File names identify the diffusion model used by an experiment. Not every
metric currently has a script for every model listed in the project roadmap;
the repository tree is the authoritative list of implemented experiments.

## Utilities

`scripts/check_environment.py` reports Python, package, CUDA, and visible-GPU
metadata for attaching to benchmark results. It does not load diffusion model
weights or start an experiment.
