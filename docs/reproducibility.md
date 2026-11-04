# Reproducibility checklist

The scripts seed Python, NumPy, and PyTorch with `42`. This improves repeatable
prompt selection and generation, but it does not by itself guarantee bitwise
identical GPU results across hardware and library versions.

## Before a run

- Record the Git commit hash with `git rev-parse HEAD`.
- Record `nvidia-smi` output and package versions.
- Record the exact Hugging Face model identifier and revision.
- Confirm dataset paths and the selected COCO subset.
- Record image count, resolution, batch sizes, and step sizes.
- Confirm that no unrelated GPU process is active.
- Decide whether generated artifacts from an older run should be reused.

## During a run

- Save the console log together with the CSV.
- Watch for fallback warnings, OOM rows, and failed metric rows.
- Avoid changing GPU clocks, power limits, or workload on the same device.
- Do not combine partial runs produced with different configurations.

## After a run

- Verify the CSV row count against the intended sweep.
- Check for `OOM`, `ERROR`, NaN, or empty values.
- Save configuration changes or a patch with the result.
- Repeat a small subset to estimate run-to-run variance.

## Determinism caveats

CUDA kernels, attention implementations, mixed precision, and package upgrades
can change results. Timing is also affected by temperature, clock state, model
cache state, and other system activity. For scientific comparisons, run each
configuration multiple times and report both central tendency and variation.
