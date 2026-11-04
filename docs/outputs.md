# Output files and analysis

## Latency CSV files

Latency scripts write one row for every `(batch size, step count)` pair. Values
may contain the string `OOM` when the configuration exceeds available memory.
Treat those cells as missing observations, not numeric zeroes.

Useful plots include:

- response time versus denoising steps, grouped by batch size;
- throughput versus batch size, grouped by step count;
- peak memory versus batch size;
- throughput versus latency to expose the batching trade-off.

## Quality CSV files

Quality scripts write metric values by step count. FID is better when lower,
LPIPS is better when lower, and CLIP similarity is generally better when
higher. Always preserve sample count and resolution columns when combining
results.

Join quality and latency tables on model and step count only when their model
revision, resolution, scheduler, precision, and prompt protocol match.

## Scheduler outputs

PaniniServe produces request-level measurements and a summary. Request-level
data is appropriate for latency distributions, deadline misses, model-switch
analysis, and per-tier comparisons. Averages alone can hide long-tail SLO
violations, so include percentiles or an empirical CDF in reports.

## Result provenance

Store a small metadata record beside each final CSV containing:

- Git commit hash;
- run date and host identifier;
- GPU, driver, CUDA, Python, PyTorch, Diffusers, and Transformers versions;
- model identifiers and revisions;
- modified configuration values;
- dataset subset and sample count.

Generated images are intentionally ignored by Git. Archive large artifacts in
experiment storage and keep only compact results and metadata in the repository
when appropriate.
