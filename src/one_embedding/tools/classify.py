"""Family classification via k-NN retrieval."""
import numpy as np
from ._base import load_protein_vecs


def predict(oemb_path, db=None, k=1, **kwargs):
    """Classify proteins by nearest neighbor in reference database.
    Returns {pid: {label, confidence, neighbors}}.
    """
    query_vecs = load_protein_vecs(oemb_path)
    db_vecs = load_protein_vecs(db) if db else query_vecs

    # Build cosine similarity
    results = {}
    db_names = sorted(db_vecs.keys())
    db_matrix = np.array([db_vecs[n] for n in db_names], dtype=np.float32)
    db_norms = np.linalg.norm(db_matrix, axis=1, keepdims=True) + 1e-10
    db_normed = db_matrix / db_norms

    for pid, vec in query_vecs.items():
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        sims = db_normed @ vec_norm
        top_k = np.argsort(-sims)[:k+1]
        neighbors = [(db_names[i], float(sims[i])) for i in top_k if db_names[i] != pid][:k]
        results[pid] = {"neighbors": neighbors, "top_match": neighbors[0] if neighbors else None}
    return results
