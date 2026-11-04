# Documentation index

The documents in this directory describe how to reproduce and interpret the
DIFS experiments. Start with setup and dataset preparation, then choose a
performance, quality, or serving experiment.

## Getting started

- [Environment setup](setup.md)
- [COCO dataset preparation](dataset.md)
- [Repository structure](project-structure.md)

## Experiments

- [Latency and throughput benchmarks](latency-benchmarks.md)
- [Image-quality metrics](quality-metrics.md)
- [PaniniServe scheduling experiment](panini-scheduler.md)

## Working with results

- [Output files and analysis](outputs.md)
- [Environment report](environment-report.md)
- [Reproducible runs](reproducibility.md)
- [Troubleshooting](troubleshooting.md)

The scripts are configured directly in Python. Read the configuration block
near the top of a script before starting a run because dataset paths, output
paths, image counts, resolutions, and sweep sizes vary by experiment.
