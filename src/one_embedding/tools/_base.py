"""Shared tool template."""
import sys
from pathlib import Path

# Ensure project root is on sys.path when tools are used standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import h5py
from src.one_embedding.io import read_oemb, read_oemb_batch


def load_per_residue(path):
    """Load per-residue embeddings from .oemb. Returns {pid: (L, D) float32}."""
    path = str(path)
    with h5py.File(path, "r") as f:
        is_batch = "per_residue" not in f
    if is_batch:
        data = read_oemb_batch(path)
    else:
        data = {"protein": read_oemb(path)}
    return {pid: d["per_residue"].astype(np.float32) for pid, d in data.items()}


def load_protein_vecs(path):
    """Load protein vectors from .oemb. Returns {pid: (V,) float32}."""
    path = str(path)
    with h5py.File(path, "r") as f:
        is_batch = "per_residue" not in f
    if is_batch:
        data = read_oemb_batch(path)
    else:
        data = {"protein": read_oemb(path)}
    return {pid: d["protein_vec"].astype(np.float32) for pid, d in data.items()}
