"""Print reproducibility metadata without starting a model download."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


PACKAGES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "numpy",
    "pandas",
    "Pillow",
    "torch-fidelity",
    "lpips",
    "nvidia-ml-py",
)


def package_version(name: str) -> str:
    """Return an installed distribution version or a readable placeholder."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def print_system_info() -> None:
    """Print Python, operating-system, and dependency information."""
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print("Packages:")
    for package in PACKAGES:
        print(f"  {package}: {package_version(package)}")


def print_cuda_info() -> None:
    """Print CUDA details when PyTorch is installed and a GPU is available."""
    try:
        import torch
    except ImportError:
        print("CUDA: unavailable (PyTorch is not installed)")
        return

    print(f"PyTorch CUDA build: {torch.version.cuda or 'none'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / 1024**3
            print(f"  GPU {index}: {properties.name} ({memory_gib:.1f} GiB)")


if __name__ == "__main__":
    print_system_info()
    print_cuda_info()
