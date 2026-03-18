"""Secondary structure prediction (3-state: H/E/C).

Uses a trained 2-layer CNN probe on 512d compressed embeddings.
Trained on CB513 (400 train / 111 test), Q3=0.432 on test set.
"""
import numpy as np
from pathlib import Path
from ._base import load_per_residue

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_MODEL_CACHE = {}


def _load_cnn():
    """Load pre-trained SS3 CNN probe. Cached after first call."""
    if "ss3" in _MODEL_CACHE:
        return _MODEL_CACHE["ss3"]

    import torch
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(512, 32, kernel_size=7, padding=3),
                nn.Tanh(),
                nn.Conv1d(32, 3, kernel_size=7, padding=3),
            )

        def forward(self, x):
            return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)

    model = CNN()
    weights_path = _WEIGHTS_DIR / "ss3_cnn_512d.pt"
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE["ss3"] = model
    return model


def predict(oemb_path, method="cnn", **kwargs):
    """Predict 3-state secondary structure per residue.

    Args:
        oemb_path: path to .oemb file
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

    # CNN probe
    import torch
    model = _load_cnn()
    results = {}
    with torch.no_grad():
        for pid, emb in embeddings.items():
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            pred = model(x).squeeze(0).argmax(dim=1).numpy()[:emb.shape[0]]
            results[pid] = pred
    return results
