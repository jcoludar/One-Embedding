"""Build the centering reference set from the UNION of PBC train splits.

Exact-id-disjoint from every PBC test split (only train ids are ever passed in); centering is
an unsupervised per-channel mean over ~2000 sequences, so residual homology influence is
negligible. Deterministic under seed.
"""
import numpy as np


def sample_reference_ids(train_ids, n=2000, seed=42):
    """Return a deterministic sample of up to n ids from the pooled train ids.

    Caps at the population size; sorted output for reproducibility.
    """
    ids = sorted(set(train_ids))
    if len(ids) <= n:
        return ids
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(idx)]
