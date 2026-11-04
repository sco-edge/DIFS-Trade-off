# Troubleshooting

## CUDA is unavailable

Check `nvidia-smi`, the installed NVIDIA driver, and whether the PyTorch build
includes CUDA support. Verify with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## A model download is denied

Open the model page on Hugging Face, accept any license terms, and run
`huggingface-cli login`. Confirm that the authenticated account can access the
exact identifier used in `from_pretrained`.

## COCO annotations are not found

Replace the checked-in `coco_annotation_path` with the absolute path to
`captions_val2014.json` on the current machine. FID scripts may also require a
separate path to validation images.

## GPU out of memory

Start with batch size 1, reduce resolution or sample count only if compatible
with the intended experiment, and close other GPU processes. Some latency
scripts record `OOM` and continue, while quality scripts record an error for
the current step setting.

## NVML initialization fails

The scheduler requires an NVIDIA driver and the `nvidia-ml-py` package. Test
that `nvidia-smi` works in the same environment. Containers also need access to
the host GPU and NVML library.

## Results are duplicated

Some quality scripts append to an existing CSV. Move or rename an earlier
result before rerunning, or verify and remove duplicate rows during analysis.
Do not delete generated reference images unless you intend to regenerate them.

## Results differ between machines

Compare GPU model, precision, model revision, scheduler, seed handling, CUDA,
PyTorch, Diffusers, attention backend, resolution, prompts, and metric sample
count. Fixed random seeds do not eliminate all GPU and software variability.
