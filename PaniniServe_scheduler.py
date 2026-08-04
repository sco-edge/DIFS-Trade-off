"""
PaniniServe -- Real GPU Execution Script (with SLO & Throughput Reporting)
============================================================================
This script takes the schedule decided by the PaniniServe scheduler
(request order, assigned model, assigned denoising steps) and actually
executes it on real hardware using Stable Diffusion v2.1 (small) and
Stable Diffusion 3.5-Medium (large), driven by real COCO2014 prompts.

Pipeline:
  1) Generate 300 requests via a Poisson arrival process, with model
     preference (small/large/auto) and accuracy level (high/medium/low)
     assigned uniformly at random to each request.
  2) Run the PaniniServe scheduler in "planning" mode (pure simulation,
     using profiled TIME_PER_STEP/BASE_OVERHEAD/SWITCH_TIME estimates)
     to decide, ahead of time, the execution order, which model each
     request will run on, and how many denoising steps it will use.
  3) Replay that plan on the real GPU: sleep until each request's
     original arrival time (so real queueing/contention is reproduced
     faithfully), then call the actual diffusers pipeline.
  4) Log one row per request to CSV (measured latency, SLO outcome,
     GPU utilization, etc.), then compute and save a summary report.
"""

import sys
import os
import json
import time
import random
import csv
import gc
import threading
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, DDIMScheduler
from pynvml import *

sys.stdout.reconfigure(line_buffering=True)

# =====================================================================
# Reproducibility
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =====================================================================
# GPU utilization monitor
# =====================================================================
# Runs on a background thread and polls GPU utilization every 10ms while
# a request is being generated, so we can report the average utilization
# for that single request's execution window.
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)


class GPUUtilMonitor:
    def __init__(self, handle):
        self.handle = handle
        self.utils = []
        self.stopped = False

    def start(self):
        self.utils = []
        self.stopped = False
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while not self.stopped:
            try:
                util = nvmlDeviceGetUtilizationRates(self.handle)
                self.utils.append(util.gpu)
            except Exception:
                pass
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        self.thread.join()
        if not self.utils:
            return 0
        return sum(self.utils) / len(self.utils)


monitor = GPUUtilMonitor(handle)

# =====================================================================
# Experiment configuration
# =====================================================================
device = "cuda"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "PaniniServe_low.csv"

TOTAL_REQUESTS = 300
LOAD_LEVEL = "low"          # "low" | "medium" | "high" -- change per run
WINDOW_SEC = 60.0             # scheduling window length, in seconds

RES = {"small": (768, 768), "large": (1024, 1024)}   # generation resolution per model

MODEL_IDS = {
    "small": "Manojb/stable-diffusion-2-1-base",             # SDv2.1
    "large": "stabilityai/stable-diffusion-3.5-medium",         # SDv3.5-Medium
}

# Denoising-step range (min, max) per requested accuracy tier
ACCURACY_STEP_RANGE = {"high": (30, 40), "medium": (20, 30), "low": (10, 20)}

# Instantaneous Poisson arrival rate range (requests/minute) per traffic load level
LOAD_LEVELS = {"low": (1.0, 2.0), "medium": (3.0, 4.0), "high": (10.0, 18.0)}

# The latency SLO multiplier is interpolated linearly across the full
# 10-40 step range, from 2.0x (at 10 steps) to 3.0x (at 40 steps), rather
# than a single fixed value -- this gives lighter (fewer-step) requests a
# tighter deadline and heavier requests more slack, both scaled off the
# same "how long would this profiled model take" baseline.
SLO_MULTIPLIER_RANGE = (2.0, 3.0)
STEP_RANGE_BOUNDS = (10, 40)

MODEL_TYPES = ["small", "large", "auto"]
ACCURACY_LEVELS = ["high", "medium", "low"]
INITIAL_MODEL = "small"   # model assumed to already be resident on the GPU at t=0

# --- Model Profile ---
# These are offline-profiled constants used ONLY by the scheduler to
# *predict* execution time when deciding the plan (order/model/steps).
# The real, measured execution time is recorded separately in the CSV
# (System_Latency_s) once the request has actually been run.
#   latency(steps) = BASE_OVERHEAD[model] + steps * TIME_PER_STEP[model]
# (BASE_OVERHEAD captures the fixed cost that doesn't scale with step
#  count -- e.g. text encoding, VAE decoding -- fitted via linear
#  regression on measured (steps, latency) pairs.)
TIME_PER_STEP = {"small": 0.08093, "large": 0.39624}   # seconds per denoising step
BASE_OVERHEAD = {"small": 0.09463, "large": 0.27221}    # fixed per-request overhead, seconds
SWITCH_TIME = {("small", "large"): 2.30, ("large", "small"): 4.50}   # model-swap overhead, seconds (asymmetric)


def exec_time(model: str, steps: float) -> float:
    """Predicted wall-clock time (seconds) to run `model` for `steps` denoising steps."""
    return BASE_OVERHEAD[model] + steps * TIME_PER_STEP[model]


def switch_cost(prev_model, next_model) -> float:
    """Predicted overhead (seconds) of swapping the GPU-resident model from
    prev_model to next_model. Zero if they're the same model, or if
    prev_model is None (i.e. this is the very first request)."""
    if prev_model is None or prev_model == next_model:
        return 0.0
    return SWITCH_TIME[(prev_model, next_model)]


def slo_multiplier(steps: float) -> float:
    """Linearly interpolate the latency-SLO multiplier across the 10-40
    step range: 2.0x at 10 steps, 3.0x at 40 steps."""
    lo, hi = STEP_RANGE_BOUNDS
    m_lo, m_hi = SLO_MULTIPLIER_RANGE
    frac = (steps - lo) / (hi - lo) if hi > lo else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return m_lo + frac * (m_hi - m_lo)


# =====================================================================
# Request data model
# =====================================================================

@dataclass
class Req:
    rid: int                  # request id
    arrival: float              # simulated arrival time (seconds since t=0)
    prompt: str                   # COCO2014 caption used as the generation prompt
    model_pref: str                 # "small" | "large" | "auto" -- what the request asked for
    accuracy: str                     # "high" | "medium" | "low" -- requested quality tier
    default_steps: int                  # max step count for this accuracy tier (starting point)
    min_steps: int                        # min step count for this accuracy tier (hard floor)
    calc_model: str                         # model used when computing this request's latency SLO
                                              # (for "auto" requests we conservatively assume "large",
                                              #  the slower model, so the deadline never ends up
                                              #  tighter than what either candidate model could meet)
    latency_slo: float                        # allowed latency budget (seconds) computed at generation time
    deadline: float                             # arrival + latency_slo -- absolute deadline

    # --- filled in later, first by the scheduler's plan, then overwritten
    #     with real measured values once the request actually executes ---
    assigned_model: Optional[str] = None
    assigned_steps: Optional[int] = None
    start: Optional[float] = None          # predicted start (from planning) -> overwritten with real value
    end: Optional[float] = None            # predicted end (from planning) -> overwritten with real value
    dropped: bool = False                    # True if the scheduler gave up on this request entirely
    switched: bool = False                     # True if a model swap happened immediately before this request
    status: Optional[str] = None                 # "met" | "soft" | "dropped"
    overage_pct: Optional[float] = None            # if "soft": % by which the deadline was exceeded

    @property
    def accuracy_met(self) -> bool:
        """Accuracy SLO is satisfied iff the request wasn't dropped AND it
        ran with at least the minimum step count its own accuracy tier
        requires. Note this is evaluated against THIS request's own
        min_steps -- never against a global/load-level threshold."""
        return (not self.dropped) and self.assigned_steps is not None \
            and self.assigned_steps >= self.min_steps


# =====================================================================
# Data loading
# =====================================================================

def load_coco_prompts(json_path: str, num_samples: int) -> List[str]:
    """Load COCO2014 caption annotations, de-duplicate, sort for a
    deterministic ordering, and take the first `num_samples`."""
    print("[*] Loading COCO prompts...")
    with open(json_path, "r") as f:
        data = json.load(f)
    captions = list(set([ann["caption"] for ann in data["annotations"]]))
    captions = sorted(captions)
    return captions[:num_samples]


def generate_requests(load_level: str, prompt_pool: List[str]) -> List[Req]:
    """Generate TOTAL_REQUESTS requests via a Poisson arrival process at the
    given load level. Each request gets a uniformly-random model preference
    and accuracy tier, and exactly one prompt from prompt_pool (shuffled so
    the prompt-to-request mapping is randomized, with no repeats)."""
    prompts = prompt_pool.copy()
    random.shuffle(prompts)

    lo_rate, hi_rate = LOAD_LEVELS[load_level]
    reqs: List[Req] = []
    t = 0.0
    for rid in range(TOTAL_REQUESTS):
        # Re-sample the instantaneous rate within the load level's band each
        # time, then draw the inter-arrival gap from an exponential
        # distribution -- this is the defining property of a Poisson process.
        rate_per_min = random.uniform(lo_rate, hi_rate)
        t += random.expovariate(rate_per_min / 60.0)

        model_pref = random.choice(MODEL_TYPES)
        accuracy = random.choice(ACCURACY_LEVELS)
        mn, mx = ACCURACY_STEP_RANGE[accuracy]

        # For "auto" requests we don't yet know which model will actually
        # run them (the scheduler decides that later), so we conservatively
        # compute the latency SLO assuming the slower "large" model -- this
        # guarantees the deadline is never tighter than either candidate
        # model could realistically meet.
        calc_model = "large" if model_pref == "auto" else model_pref
        latency_slo = exec_time(calc_model, mx) * slo_multiplier(mx)
        deadline = t + latency_slo

        reqs.append(Req(
            rid=rid, arrival=round(t, 3), prompt=prompts[rid],
            model_pref=model_pref, accuracy=accuracy,
            default_steps=mx, min_steps=mn, calc_model=calc_model,
            latency_slo=round(latency_slo, 3), deadline=round(deadline, 3),
        ))
    return reqs


# =====================================================================
# PaniniServe scheduler (planning / simulation phase)
# =====================================================================
# Everything in this section only PREDICTS timing using the Model Profile
# constants above -- it never touches the GPU. Its job is to decide, for
# every request: which order to run them in, which model each one uses,
# and how many steps each one gets, before any real generation happens.

def order_batch(batch: List[Req], current_model: str) -> List[Req]:
    """Order one window's worth of requests (EDF + locality-aware switch
    minimization):
      1) Requests that already want the currently-loaded model go first,
         sorted by deadline (Earliest Deadline First).
      2) Requests that want the other model go next, also sorted by
         deadline (so the whole queue only pays for ONE model switch, at
         the boundary between these two blocks).
      3) "auto" requests (model not yet decided) are inserted one at a
         time, most-urgent-deadline first, into the latest position that
         still lets them meet their own latency_slo -- and they inherit
         whichever model is adjacent at that position, so they never
         introduce an extra switch.
    """
    front = [r for r in batch if r.model_pref == current_model]
    other_model = "large" if current_model == "small" else "small"
    back = [r for r in batch if r.model_pref == other_model]
    auto_reqs = [r for r in batch if r.model_pref == "auto"]
    front.sort(key=lambda r: r.deadline)
    back.sort(key=lambda r: r.deadline)
    for r in front:
        r.assigned_model = current_model
    for r in back:
        r.assigned_model = other_model
    queue = front + back
    for req in sorted(auto_reqs, key=lambda r: r.deadline):
        best_pos = None
        # Scan from the back of the queue forward, looking for the latest
        # (least disruptive) position where inserting this request still
        # meets its own latency_slo.
        for pos in range(len(queue), -1, -1):
            left = queue[pos - 1].assigned_model if pos > 0 else current_model
            right = queue[pos].assigned_model if pos < len(queue) else None
            req.assigned_model = left or right or current_model   # inherit neighbor's model -> no new switch
            t = 0.0
            prev = current_model
            for r in queue[:pos] + [req]:
                t += switch_cost(prev, r.assigned_model)
                t += exec_time(r.assigned_model, r.default_steps)
                prev = r.assigned_model
            if t <= req.latency_slo:
                best_pos = pos
                break
        if best_pos is None:
            # No position satisfies the deadline -- fall back to appending
            # at the very end (still inheriting the last model, so at least
            # no extra switch is introduced).
            best_pos = len(queue)
            req.assigned_model = queue[-1].assigned_model if queue else current_model
        queue = queue[:best_pos] + [req] + queue[best_pos:]
    return queue


def total_window_time(queue, current_model, use_assigned=False) -> float:
    """Predicted total time (including model-switch overhead) to run the
    entire queue in order, starting from current_model.
    use_assigned=False uses each request's default (max) step count --
    used for feasibility checks before steps have been finalized.
    use_assigned=True uses whatever assigned_steps has actually been set --
    used once steps have been decided (possibly reduced)."""
    t, prev = 0.0, current_model
    for r in queue:
        t += switch_cost(prev, r.assigned_model)
        steps = r.assigned_steps if (use_assigned and r.assigned_steps is not None) else r.default_steps
        t += exec_time(r.assigned_model, steps)
        prev = r.assigned_model
    return t


def plan_schedule(reqs: List[Req]) -> List[Req]:
    """Run the full PaniniServe windowed scheduling loop in simulation mode:
    decide order/model/steps/drops for every request, without touching the
    GPU. The returned list is sorted into actual execution order and is
    ready to be replayed for real by run_real_execution().

    Per window:
      1) Collect all requests that have arrived and whose deadline falls
         within this window.
      2) Order them (EDF + locality-aware, see order_batch()).
      3) If the predicted total time exceeds the window's available time,
         shrink each request's step count proportionally (fair-share),
         never going below that request's accuracy-tier minimum.
      4) If it still doesn't fit, drop requests starting from the latest
         deadline until it does.
    """
    pending = sorted(reqs, key=lambda r: r.arrival)
    waiting: List[Req] = []
    current_model = INITIAL_MODEL
    window_start, current_time = 0.0, 0.0
    planned: List[Req] = []

    while len(planned) < len(reqs):
        window_end = window_start + WINDOW_SEC
        arrived = [r for r in pending if r.arrival <= window_end]
        for r in arrived:
            waiting.append(r)
            pending.remove(r)
        if not waiting:
            window_start = window_end
            current_time = max(current_time, window_start)
            continue

        # "Include every request whose deadline falls inside this window"
        batch = [r for r in waiting if r.deadline <= window_end]
        if not batch:
            # Nothing has a deadline in this window, but don't idle --
            # process whichever waiting request is most urgent.
            batch = [min(waiting, key=lambda r: r.deadline)]

        queue = order_batch(batch, current_model)
        for r in queue:
            r.assigned_steps = r.default_steps   # start optimistic: full (max) step count

        start_t = max(current_time, window_start)
        max_allowed = window_end - start_t
        if max_allowed <= 0:
            # The GPU was still busy past this window's end -- there's no
            # time left in this window at all, so drop everything queued.
            for r in queue:
                r.dropped, r.status = True, "dropped"
                if r in waiting:
                    waiting.remove(r)
                planned.append(r)
            window_start = window_end
            continue

        total = total_window_time(queue, current_model, use_assigned=True)

        # Step 1: if we're over budget, shrink steps fair-share style
        if total > max_allowed:
            overflow = total - max_allowed
            reduce_each = overflow / len(queue)   # split the overflow evenly across requests
            for r in queue:
                step_reduction = math.floor(reduce_each / TIME_PER_STEP[r.assigned_model])
                r.assigned_steps = max(r.default_steps - step_reduction, r.min_steps)   # never below the floor
            total = total_window_time(queue, current_model, use_assigned=True)

        # Step 2: if STILL over budget even at minimum steps, drop the
        # latest-deadline requests one at a time until it fits.
        if total > max_allowed:
            for r in sorted(queue, key=lambda r: -r.deadline):
                if total <= max_allowed:
                    break
                queue.remove(r)
                if r in waiting:
                    waiting.remove(r)
                r.dropped, r.status = True, "dropped"
                planned.append(r)
                total = total_window_time(queue, current_model, use_assigned=True)

        # Record the PREDICTED start/end for each request (these will be
        # overwritten with real measured values during actual execution).
        t, prev = start_t, current_model
        for r in queue:
            sc = switch_cost(prev, r.assigned_model)
            r.switched = sc > 0
            t += sc
            r.start = t
            t += exec_time(r.assigned_model, r.assigned_steps)
            r.end = t
            prev = r.assigned_model

        if queue:
            current_model = queue[-1].assigned_model   # remember which model is loaded going into the next window
            current_time = t
        for r in queue:
            if r in waiting:
                waiting.remove(r)
        planned.extend(queue)
        window_start = window_end

    # Sort into the order requests will actually be executed in (dropped
    # requests last, everything else by predicted start time).
    planned.sort(key=lambda r: (r.dropped, r.start if r.start is not None else r.arrival))
    return planned


# =====================================================================
# Model loading & warm-up
# =====================================================================

def load_pipelines():
    """Load both diffusion pipelines. Both start out on CPU; switch_to()
    moves whichever one is needed onto the GPU on demand."""
    print("[*] Loading SD v2.1 (768x768)...")
    pipe_small = StableDiffusionPipeline.from_pretrained(
        MODEL_IDS["small"], torch_dtype=torch.float16, safety_checker=None
    )
    pipe_small.scheduler = DDIMScheduler.from_config(pipe_small.scheduler.config)
    pipe_small.enable_attention_slicing()
    try:
        pipe_small.enable_xformers_memory_efficient_attention()
        print("[*] xformers ON (SDv2.1)")
    except Exception:
        print("[!] xformers not available (SDv2.1)")
    pipe_small.set_progress_bar_config(disable=True)

    print("[*] Loading SD3.5 Medium (1024x1024)...")
    pipe_large = StableDiffusion3Pipeline.from_pretrained(
        MODEL_IDS["large"], torch_dtype=torch.bfloat16
    )
    pipe_large.set_progress_bar_config(disable=True)

    return {"small": pipe_small, "large": pipe_large}


def switch_to(pipes: dict, model_name: str, current_on_gpu: Optional[str]) -> float:
    """Move `model_name` onto the GPU (a no-op if it's already there), and
    move whichever OTHER model was previously on the GPU back to CPU to
    free VRAM. Returns the real, measured time (seconds) this swap took --
    this is how the SWITCH_TIME profile constants above were originally
    obtained, and it's also logged per-request so you can validate the
    profile against fresh measurements.
    `silence_dtype_warnings=True` suppresses PyTorch's (harmless, in our
    case) warning about moving a float16 pipeline to CPU -- we never run
    inference on the CPU-resident copy, so this is safe to ignore."""
    if current_on_gpu == model_name:
        return 0.0
    t0 = time.time()
    if current_on_gpu is not None:
        pipes[current_on_gpu].to("cpu", silence_dtype_warnings=True)
    pipes[model_name].to(device)
    torch.cuda.synchronize()
    return time.time() - t0


def warm_up(pipes: dict, prompt_pool: List[str]):
    """Run a small throwaway generation on each model before the real
    experiment starts, so that one-time costs (CUDA kernel compilation,
    memory allocator warm-up, etc.) don't pollute the first few real
    measurements."""
    print("[*] Warming up...")
    switch_to(pipes, "small", None)
    with torch.inference_mode():
        H, W = RES["small"]
        _ = pipes["small"](prompt_pool[:2], num_inference_steps=20, height=H, width=W)
    torch.cuda.synchronize()

    switch_to(pipes, "large", "small")
    with torch.inference_mode():
        H, W = RES["large"]
        _ = pipes["large"](prompt_pool[:2], num_inference_steps=20, height=H, width=W)
    torch.cuda.synchronize()

    print("[*] Warm-up complete!\n")


# =====================================================================
# Real GPU execution (real-time replay + full reporting)
# =====================================================================

def run_real_execution(plan: List[Req], pipes: dict, realtime: bool = True,
                       save_dir: Optional[str] = None):
    """Replay the schedule produced by plan_schedule() for real.

    realtime=True  : sleep until each request's original simulated arrival
                      time before running it, so real queueing/contention
                      under the target load level is faithfully reproduced.
                      This is the mode to use for numbers that will be
                      reported in the paper.
    realtime=False : run every (non-dropped) request back-to-back with no
                      waiting -- useful only for quickly sanity-checking
                      that the Model Profile constants roughly match real
                      hardware; the resulting SLO numbers are NOT
                      representative of real load and should not be
                      reported as such.

    Dropped requests are skipped (the scheduler already decided not to run
    them) but still get a row in the CSV so they count correctly toward
    the denominator when computing SLO attainment rates.
    """
    current_on_gpu = None
    t0 = time.time()

    with open(csv_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "RID", "Model", "Steps", "Arrival_s", "Deadline_s",
            "Real_Start_s", "Real_End_s", "System_Latency_s",
            "Switch_Time_s", "Status", "Latency_SLO_Met", "Accuracy_SLO_Met",
            "Overage_pct", "Peak_Mem_GB", "GPU_Util_%"
        ])

    print(f"{'RID':<5} | {'Model':<6} | {'Steps':<5} | {'System_Lat_s':<12} | {'Switch_s':<9} | {'Status':<8} | {'GPU%':<6}")
    print("-" * 75)

    for r in plan:
        if r.dropped:
            # Log a placeholder row so drops are still counted in the
            # summary statistics, even though nothing was actually run.
            with open(csv_output_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    r.rid, r.assigned_model, r.assigned_steps,
                    r.arrival, r.deadline, "", "", "", "",
                    "dropped", False, False, "", "", ""
                ])
            continue

        if realtime:
            # Wait until wall-clock time matches this request's original
            # simulated arrival time, so real queueing delay accumulates
            # exactly as it would under real production traffic.
            wait = r.arrival - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

        # Reset memory bookkeeping so Peak_Mem_GB reflects only this request.
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        monitor.start()
        switch_time = switch_to(pipes, r.assigned_model, current_on_gpu)
        current_on_gpu = r.assigned_model

        real_start = time.time() - t0
        H, W = RES[r.assigned_model]
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        with torch.inference_mode():
            image = pipes[r.assigned_model](
                r.prompt, num_inference_steps=r.assigned_steps,
                height=H, width=W, generator=generator,
            ).images[0]
        torch.cuda.synchronize()
        real_end = time.time() - t0
        gpu_util = monitor.stop()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        # Overwrite the scheduler's PREDICTED start/end with the REAL
        # measured values, and re-evaluate SLO status against reality.
        r.start, r.end = real_start, real_end

        # --- metrics ---
        system_latency = real_end - r.arrival   # end-to-end latency, including queueing wait
        latency_slo_met = r.end <= r.deadline    # Latency SLO: only "met" counts as success

        # Accuracy SLO must be judged against THIS request's own
        # accuracy-tier floor (r.min_steps), never against LOAD_LEVEL --
        # LOAD_LEVEL is a traffic-load setting for the whole run, not a
        # per-request accuracy setting, so keying a threshold off it would
        # apply the same floor to every request regardless of what tier it
        # actually asked for. r.accuracy_met already implements the
        # correct per-request check.
        accuracy_slo_met = r.accuracy_met   # == r.assigned_steps >= r.min_steps

        r.status = "met" if latency_slo_met else "soft"
        if r.status == "soft":
            r.overage_pct = (r.end - r.deadline) / r.latency_slo * 100.0

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            image.save(os.path.join(save_dir, f"req_{r.rid:03d}.png"))

        print(f"{r.rid:<5} | {r.assigned_model:<6} | {r.assigned_steps:<5} | "
              f"{system_latency:<12.4f} | {switch_time:<9.4f} | {r.status:<8} | {gpu_util:<6.1f}")

        with open(csv_output_file, "a", newline="") as f:
            csv.writer(f).writerow([
                r.rid, r.assigned_model, r.assigned_steps, r.arrival, r.deadline,
                real_start, real_end, system_latency, switch_time,
                r.status, latency_slo_met, accuracy_slo_met,
                r.overage_pct or "", peak_mem, gpu_util
            ])

    exp_total_time = time.time() - t0
    print(f"\n[OK] Run complete -> {os.path.abspath(csv_output_file)}")

    # =====================================================================
    # Final aggregate report: console summary + summary CSV
    # =====================================================================
    df = pd.read_csv(csv_output_file)
    total_reqs = len(df)
    completed_reqs = len(df[df["Status"] != "dropped"])

    # 1. Throughput -- completed (non-dropped) requests per second of
    #    total wall-clock experiment time
    req_throughput = completed_reqs / exp_total_time

    # 2. Latency SLO -- only "met" counts; "soft" (late-but-completed) does
    #    NOT count as success here
    latency_met_cnt = (df["Latency_SLO_Met"] == True).sum()
    latency_slo_rate = (latency_met_cnt / total_reqs) * 100
    p50_lat = df["System_Latency_s"].dropna().quantile(0.50)
    p95_lat = df["System_Latency_s"].dropna().quantile(0.95)
    p99_lat = df["System_Latency_s"].dropna().quantile(0.99)

    # 3. Accuracy SLO -- dropped requests always fail this; everything else
    #    passes as long as it ran at or above its own tier's minimum steps
    #    (by construction, PaniniServe's step-reduction logic never goes
    #    below that floor, so in practice Accuracy_SLO_Attainment ==
    #    100% - Drop_rate here -- this is expected, not a bug)
    acc_met_cnt = (df["Accuracy_SLO_Met"] == True).sum()
    acc_slo_rate = (acc_met_cnt / total_reqs) * 100

    # --- console summary ---
    print("\n" + "=" * 65)
    print(f"       PaniniServe Performance Summary Report ({LOAD_LEVEL.upper()})")
    print("=" * 65)
    print(f" [Throughput]")
    print(f"  - Total Elapsed Time     : {exp_total_time:.2f} s")
    print(f"  - Request Throughput     : {req_throughput:.3f} req/s")
    print("-" * 65)
    print(f" [Latency SLO]")
    print(f"  - Latency SLO Attainment : {latency_slo_rate:.2f}% ({latency_met_cnt}/{total_reqs})")
    print(f"  - System Latency P50     : {p50_lat:.3f} s")
    print(f"  - System Latency P95     : {p95_lat:.3f} s")
    print(f"  - System Latency P99     : {p99_lat:.3f} s")
    print("-" * 65)
    print(f" [Accuracy SLO]")
    print(f"  - Accuracy SLO Attainment: {acc_slo_rate:.2f}% ({acc_met_cnt}/{total_reqs})")
    print("=" * 65 + "\n")

    # --- summary CSV ---
    summary_csv_file = f"PaniniServe_summary_{LOAD_LEVEL}.csv"
    summary_data = {
        "Metric": [
            "Load_Level", "Total_Requests", "Completed_Requests", "Total_Elapsed_Time_s",
            "Throughput_req_s", "Latency_SLO_Attainment_%", "System_Latency_P50_s",
            "System_Latency_P95_s", "System_Latency_P99_s", "Accuracy_SLO_Attainment_%"
        ],
        "Value": [
            LOAD_LEVEL, total_reqs, completed_reqs, round(exp_total_time, 2),
            round(req_throughput, 3), round(latency_slo_rate, 2), round(p50_lat, 3),
            round(p95_lat, 3), round(p99_lat, 3), round(acc_slo_rate, 2)
        ]
    }
    pd.DataFrame(summary_data).to_csv(summary_csv_file, index=False)
    print(f"[OK] Summary report saved -> {os.path.abspath(summary_csv_file)}")

# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_REQUESTS)
    reqs = generate_requests(LOAD_LEVEL, prompt_pool)
    plan = plan_schedule(reqs)   # (1) decide the full schedule via simulation first

    pipes = load_pipelines()
    warm_up(pipes, prompt_pool)

    # Real results for the paper (sleeps to match original arrival times,
    # so real production-like load/queueing is reproduced):
    run_real_execution(plan, pipes, realtime=True, save_dir=f"outputs/panini_{LOAD_LEVEL}")
