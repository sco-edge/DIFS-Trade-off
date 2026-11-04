# Environment report

Benchmark results are only comparable when their execution environments are
known. The repository includes a read-only helper that prints the most useful
software and GPU metadata without loading or downloading a diffusion model.

Run it from the repository root:

```bash
python scripts/check_environment.py
```

The report contains the Python and operating-system versions, versions of the
project's main Python packages, the CUDA version used to build PyTorch, CUDA
availability, and the name and memory capacity of each visible GPU.

Save the report beside a benchmark CSV:

```bash
python scripts/check_environment.py > environment.txt
git rev-parse HEAD >> environment.txt
```

The Git commit is intentionally collected separately because the environment
script also works when the source directory is copied without `.git` metadata.

For a complete performance record, also retain `nvidia-smi` output. It includes
the NVIDIA driver version and may show other processes using GPU memory at the
time of collection.

Environment metadata explains many differences but does not guarantee equal
results. Also preserve model revisions, dataset subset, resolution, precision,
scheduler, batch sizes, denoising steps, seed policy, and any local code or
configuration changes.
