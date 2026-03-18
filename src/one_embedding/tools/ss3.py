"""Secondary structure prediction (3-state: H/E/C)."""
import numpy as np
from ._base import load_per_residue


def predict(oemb_path, **kwargs):
    """Predict 3-state secondary structure per residue.
    Returns {pid: (L,) array of 0=H, 1=E, 2=C}.
    """
    embeddings = load_per_residue(oemb_path)
    results = {}
    for pid, emb in embeddings.items():
        # Simple heuristic: use embedding variance as a proxy
        # (helices have lower per-residue variance than coils)
        local_var = np.var(emb, axis=1)
        # Discretize into 3 classes by terciles
        t1, t2 = np.percentile(local_var, [33, 66])
        ss = np.where(local_var < t1, 0, np.where(local_var < t2, 1, 2))
        results[pid] = ss
    return results
