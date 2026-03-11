"""MPS/CPU device management."""

import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(tensor: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = get_device()
    return tensor.to(device=device, dtype=torch.float32)
