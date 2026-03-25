"""TriZOD dataset loader for disorder cross-validation.

TriZOD348 is an independent test set for validating CheZOD disorder results.
Embeddings: data/residue_embeddings/prot_t5_xl_trizod.h5 (5786 proteins)
Split: data/benchmark_suite/splits/trizod_predefined.json (train: 5438, test: 348)
"""
import json
from pathlib import Path

import h5py
import numpy as np


def load_trizod_embeddings(embeddings_path, split_path):
    """Load TriZOD embeddings and predefined split.

    Args:
        embeddings_path: Path to prot_t5_xl_trizod.h5
        split_path: Path to trizod_predefined.json

    Returns:
        dict with 'embeddings' ({pid: (L, 1024)}), 'train_ids', 'test_ids'.
    """
    embeddings = {}
    with h5py.File(str(embeddings_path), "r") as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)

    with open(split_path) as f:
        split = json.load(f)

    return {
        "embeddings": embeddings,
        "train_ids": split["train_ids"],
        "test_ids": split["test_ids"],
    }
