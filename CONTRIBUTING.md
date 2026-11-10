# Contributing

## Keep experiments reviewable

Make one coherent change per commit. Examples include adding one model
benchmark, correcting one metric implementation, documenting one experiment,
or adding one validation check. Use commit messages that state the observable
change, such as `docs: explain FID dataset preparation`.

## Adding or changing an experiment

Document the following in the script or accompanying guide:

- model identifier and revision;
- dataset subset and sample count;
- image resolution, precision, and scheduler;
- batch-size and denoising-step sweep;
- seed policy and warm-up procedure;
- output schema and metric direction;
- expected GPU and disk requirements.

Prefer configurable paths over machine-specific absolute paths. Keep generated
images, model weights, caches, and large result directories out of Git.

## Validation

Before committing a Python change:

```bash
python -m compileall accuracy_est latency_est PaniniServe_scheduler.py
```

For GPU experiments, first run a reduced smoke-test configuration. Inspect the
CSV header and one data row, then restore or explicitly record the intended
full configuration.

## Results and claims

Do not compare measurements from incompatible hardware or protocols without
labeling the difference. Include run metadata and note OOM or failed metric
settings. A chart should state whether higher or lower metric values are
preferred and identify the model, resolution, and sample count.

## Pull request checklist

Before submitting a pull request:

- Create the branch from the latest `main`.
- Keep the pull request focused on one topic.
- Do not include generated images, model weights, caches, or result CSV files.
- Remove credentials and review machine-specific paths.
- Run `python -m compileall` when Python files are changed.
- Run a reduced smoke test when suitable GPU hardware is available.
- State clearly when runtime tests were not performed.
- Update documentation when configuration or output columns change.
- Include a summary and testing section in the pull request description.

Documentation-only pull requests do not require GPU testing.
