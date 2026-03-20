"""Shared tool template."""
import sys
from pathlib import Path

# Ensure project root is on sys.path when tools are used standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import h5py
from src.one_embedding.io import (
    read_oemb, read_oemb_batch,
    read_one_h5, read_one_h5_batch,
)


def _is_one_h5(path):
    """Check if a file is in .one.h5 format (has format='one_embedding' attr)."""
    with h5py.File(str(path), "r") as f:
        fmt = f.attrs.get("format", "")
        if isinstance(fmt, bytes):
            fmt = fmt.decode("utf-8")
        return fmt == "one_embedding"


def load_per_residue(path):
    """Load per-residue embeddings from .oemb or .one.h5. Returns {pid: (L, D) float32}."""
    path = str(path)

    # Check for .one.h5 format first
    if _is_one_h5(path):
        with h5py.File(path, "r") as f:
            n_proteins = int(f.attrs.get("n_proteins", 1))
        if n_proteins == 1:
            data = read_one_h5(path)
            pid = data.get("protein_id", "protein")
            return {pid: data["per_residue"].astype(np.float32)}
        else:
            batch = read_one_h5_batch(path)
            return {pid: d["per_residue"].astype(np.float32) for pid, d in batch.items()}

    # Legacy .oemb format
    with h5py.File(path, "r") as f:
        is_batch = "per_residue" not in f
    if is_batch:
        data = read_oemb_batch(path)
    else:
        data = {"protein": read_oemb(path)}
    return {pid: d["per_residue"].astype(np.float32) for pid, d in data.items()}


def load_protein_vecs(path):
    """Load protein vectors from .oemb or .one.h5. Returns {pid: (V,) float32}."""
    path = str(path)

    # Check for .one.h5 format first
    if _is_one_h5(path):
        with h5py.File(path, "r") as f:
            n_proteins = int(f.attrs.get("n_proteins", 1))
        if n_proteins == 1:
            data = read_one_h5(path)
            pid = data.get("protein_id", "protein")
            return {pid: data["protein_vec"].astype(np.float32)}
        else:
            batch = read_one_h5_batch(path)
            return {pid: d["protein_vec"].astype(np.float32) for pid, d in batch.items()}

    # Legacy .oemb format
    with h5py.File(path, "r") as f:
        is_batch = "per_residue" not in f
    if is_batch:
        data = read_oemb_batch(path)
    else:
        data = {"protein": read_oemb(path)}
    return {pid: d["protein_vec"].astype(np.float32) for pid, d in data.items()}
