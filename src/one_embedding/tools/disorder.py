"""Disorder prediction from compressed embeddings.

Uses a trained 2-layer CNN probe (SETH-style) on 512d compressed embeddings.
Predicts CheZOD Z-scores; lower = more disordered.
Trained on CheZOD 1174, validated ρ=0.707 on CheZOD 117.
"""
import numpy as np
from pathlib import Path
from ._base import load_per_residue

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_MODEL_CACHE = {}


def _load_cnn():
    """Load pre-trained CNN probe. Cached after first call."""
    if "disorder" in _MODEL_CACHE:
        return _MODEL_CACHE["disorder"]

    import torch
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(512, 32, kernel_size=7, padding=3),
                nn.Tanh(),
                nn.Conv1d(32, 1, kernel_size=7, padding=3),
            )

        def forward(self, x):
            return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)

    model = CNN()
    weights_path = _WEIGHTS_DIR / "disorder_cnn_512d.pt"
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE["disorder"] = model
    return model


def predict(oemb_path, method="cnn", **kwargs):
    """Predict per-residue disorder scores.

    Args:
        oemb_path: path to .oemb file
        method: "cnn" (trained, recommended) or "norm" (fast heuristic)

    Returns:
        {protein_id: np.ndarray of shape (L,)} — CheZOD Z-scores (lower = disordered)
    """
    embeddings = load_per_residue(oemb_path)

    if method == "norm":
        results = {}
        for pid, emb in embeddings.items():
            norms = np.linalg.norm(emb, axis=1)
            results[pid] = 1.0 - (norms - norms.min()) / (norms.max() - norms.min() + 1e-10)
        return results

    # CNN probe
    import torch
    model = _load_cnn()
    results = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            pred = model(x).squeeze().numpy()
            results[pid] = pred[:emb.shape[0]]
    return results
