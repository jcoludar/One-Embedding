"""Secondary structure prediction (3-state: H/E/C).

Uses a trained 2-layer CNN probe on compressed embeddings.
Trained on CB513 (408 train / 103 test, seed=42), Q3=0.855 on test set (512d).

Supports multiple embedding dimensions (512d, 768d) via auto-detection.
Falls back to heuristic if no trained weights exist for the detected dimension.
"""
import warnings

import numpy as np
from pathlib import Path
from ._base import load_per_residue

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_MODEL_CACHE = {}

# Mapping from embedding dimension to weight file name
_WEIGHT_FILES = {
    512: "ss3_cnn_512d.pt",
    768: "ss3_cnn_768d.pt",
}


def _load_cnn(input_dim=512):
    """Load pre-trained SS3 CNN probe for the given dimension. Cached after first call.

    Args:
        input_dim: Embedding dimension (e.g., 512, 768).

    Returns:
        Loaded CNN model, or None if weights not found.
    """
    cache_key = f"ss3_{input_dim}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    import torch
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(d_in, 32, kernel_size=7, padding=3),
                nn.Tanh(),
                nn.Conv1d(32, 3, kernel_size=7, padding=3),
            )

        def forward(self, x):
            return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)

    # Determine weight file
    weight_file = _WEIGHT_FILES.get(input_dim)
    if weight_file is None:
        warnings.warn(
            f"No SS3 CNN weights registered for input_dim={input_dim}. "
            f"Supported dimensions: {sorted(_WEIGHT_FILES.keys())}. "
            f"Falling back to heuristic.",
            UserWarning,
            stacklevel=3,
        )
        return None

    weights_path = _WEIGHTS_DIR / weight_file
    if not weights_path.exists():
        warnings.warn(
            f"SS3 CNN weights not found at {weights_path}. "
            f"Falling back to heuristic.",
            UserWarning,
            stacklevel=3,
        )
        return None

    model = CNN(input_dim)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def predict(oemb_path, method="cnn", **kwargs):
    """Predict 3-state secondary structure per residue.

    Auto-detects embedding dimension from the data and loads matching
    CNN weights. If no weights exist for the detected dimension, falls
    back to heuristic with a warning.

    Args:
        oemb_path: path to .oemb or .one.h5 file
        method: "cnn" (trained, recommended) or "heuristic" (fast, untrained)

    Returns:
        {pid: (L,) array of 0=H, 1=E, 2=C}
    """
    embeddings = load_per_residue(oemb_path)

    if method == "heuristic":
        results = {}
        for pid, emb in embeddings.items():
            local_var = np.var(emb, axis=1)
            t1, t2 = np.percentile(local_var, [33, 66])
            results[pid] = np.where(local_var < t1, 0, np.where(local_var < t2, 1, 2))
        return results

    # CNN probe — auto-detect dimension from the first protein
    first_emb = next(iter(embeddings.values()))
    input_dim = first_emb.shape[1]

    import torch
    model = _load_cnn(input_dim=input_dim)
    if model is None:
        # Graceful degradation: fall back to heuristic
        return predict(oemb_path, method="heuristic", **kwargs)

    results = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            pred = model(x).squeeze(0).argmax(dim=1).numpy()[:emb.shape[0]]
            results[pid] = pred
    return results
