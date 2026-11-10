# Latency and throughput benchmarks

The scripts in `latency_est/` measure a single batch's response time across a
grid of batch sizes and denoising-step counts.

## Experiment grid

The checked-in scripts use these default values:

```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 80, 96, 128]
step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
```

Large batches will not fit on many GPUs. On an out-of-memory error, a script
records `OOM` and skips larger batches for that step count.

## Run a benchmark

After setting `coco_annotation_path`, run one model script from the repository
root:

```bash
python latency_est/SD_v1.5.py
python latency_est/SD_v2.1.py
python latency_est/SDXL.py
python latency_est/SD_v3.5.py
```

Each script loads and warms up the pipeline before timing. CUDA synchronization
surrounds the measured section so asynchronous GPU work is included.

## CSV columns

| Column | Meaning |
| --- | --- |
| `Batch` | Number of prompts generated together |
| `Steps` | Denoising steps per image |
| `User_Latency_s` | Wall-clock response time for the batch |
| `Throughput_img_s` | Batch size divided by response time |
| `Peak_Mem_GB` | Peak PyTorch-allocated GPU memory |

Peak allocated memory is not the same as total process or device memory. Use
the same measurement method when comparing runs.

## Fair comparisons

Keep the GPU model, precision, resolution, scheduler, library versions, warm-up
procedure, and prompt count constant. Report batch latency and throughput
together: a configuration can improve throughput while increasing the delay
seen by one request.

## Available model benchmarks

| Model | Script | Resolution | Precision | Output CSV |
| --- | --- | --- | --- | --- |
| Stable Diffusion 1.5 | `latency_est/SD_v1.5.py` | 512x512 | float16 | `SD_v1.5_scaling.csv` |
| Stable Diffusion 2.1 | `latency_est/SD_v2.1.py` | 768x768 | float16 | `SD_v2.1_scaling.csv` |
| SDXL 1.0 | `latency_est/SDXL.py` | 1024x1024 | float32 | `SDXL_scaling.csv` |
| Stable Diffusion 3.5 Medium | `latency_est/SD_v3.5.py` | 1024x1024 | bfloat16 | `SD3.5_scaling.csv` |

Precision and resolution affect latency, memory usage, and throughput.
Modified configurations should be reported with benchmark results.
