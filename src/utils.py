
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> dict:
    """Return basic device information."""
    device = get_device()

    info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info
