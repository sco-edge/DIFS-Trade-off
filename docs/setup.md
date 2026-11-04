# Environment setup

## Hardware and system requirements

The experiments assume an NVIDIA GPU and a working CUDA-enabled PyTorch
installation. Memory requirements differ substantially by model, resolution,
batch size, and precision. Start with batch size 1 when testing a new machine.

Python 3.10 or 3.11 is a practical choice for the current PyTorch and
Diffusers ecosystem. Use a fresh virtual environment to keep CUDA-related
package conflicts isolated.

## Create an environment

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch build that matches the machine's CUDA setup, then install
the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Model access

Diffusers downloads weights from Hugging Face during the first run. If a model
is gated, accept its license on the model page and authenticate locally:

```bash
huggingface-cli login
```

The model cache can require many gigabytes. Verify available disk space before
running several model families.

## Verify the GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The first value should be `True`. The scheduler also reads utilization through
NVML, so the NVIDIA driver and `nvidia-ml-py` package must be available.

## Before the first full run

Update the dataset and output paths in the selected script. Reduce the image
count and sweep lists for a smoke test, confirm that a CSV is produced, and
only then restore the full experiment configuration.
