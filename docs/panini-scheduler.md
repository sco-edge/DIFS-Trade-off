# PaniniServe scheduler experiment

`PaniniServe_scheduler.py` plans a mixed-model workload and replays the plan on
a real GPU. It combines Stable Diffusion 2.1 as the `small` model with Stable
Diffusion 3.5 Medium as the `large` model.

## Workflow

1. Load deterministic COCO captions.
2. Generate requests from a Poisson arrival process.
3. Assign each request a model preference and quality tier.
4. Group requests in scheduling windows.
5. Order requests by deadline while reducing model switches.
6. Reduce denoising steps or drop work when a window is infeasible.
7. Replay the plan with real pipelines and record measured outcomes.

## Main configuration

| Setting | Purpose |
| --- | --- |
| `TOTAL_REQUESTS` | Number of generated requests |
| `LOAD_LEVEL` | `low`, `medium`, or `high` arrival-rate range |
| `WINDOW_SEC` | Scheduling-window duration |
| `ACCURACY_STEP_RANGE` | Minimum and default steps for each quality tier |
| `TIME_PER_STEP` | Profiled seconds per denoising step |
| `BASE_OVERHEAD` | Profiled fixed request overhead |
| `SWITCH_TIME` | Profiled asymmetric model-swap overhead |

The profile constants influence planning but are not the final measured
latency. Re-profile them when moving to a different GPU or software stack.

## Run

Update the COCO path and choose a load level, then execute:

```bash
python PaniniServe_scheduler.py
```

The experiment writes request-level CSV data, a load-specific summary CSV, and
generated images below `outputs/`. It initializes NVML at import time and
therefore requires a working NVIDIA driver.

## Reading outcomes

Requests may be marked `met`, `soft`, or `dropped`. The scheduler separately
tracks whether the assigned step count satisfies the request's accuracy floor.
Use both latency-SLO and accuracy-SLO fields when comparing scheduling policies.
