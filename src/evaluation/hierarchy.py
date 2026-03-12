"""SCOP hierarchy-aware evaluation of protein embedding distances.

Evaluates whether embedding distances respect the SCOP structural hierarchy:
  within-family < same-superfamily < same-fold < unrelated

Also generates distance distribution visualizations.
"""

import numpy as np
from scipy.spatial.distance import cdist


def evaluate_hierarchy_distances(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    metric: str = "cosine",
    max_pairs_per_level: int = 50_000,
    seed: int = 42,
) -> dict:
    """Compute mean embedding distances at each SCOP hierarchy level.

    Hierarchy levels:
        - within_family: same family
        - same_superfamily: same superfamily, different family
        - same_fold: same fold, different superfamily
        - unrelated: different folds

    Args:
        vectors: {protein_id: embedding_vector}.
        metadata: List of dicts with id, family, superfamily, fold.
        metric: "cosine" or "euclidean".
        max_pairs_per_level: Cap on pairs sampled per level (for speed).
        seed: Random seed for subsampling.

    Returns:
        Dict with mean_distance at each level, separation_ratio, counts,
        and ordering_correct bool.
    """
    rng = np.random.RandomState(seed)

    # Build lookup
    id_to_meta = {}
    for m in metadata:
        pid = m["id"]
        if pid in vectors and all(k in m for k in ("family", "superfamily", "fold")):
            id_to_meta[pid] = m

    pids = list(id_to_meta.keys())
    n = len(pids)
    if n < 10:
        return {"error": "Too few proteins with full hierarchy metadata"}

    # Build matrix
    matrix = np.array([vectors[pid] for pid in pids], dtype=np.float32)

    # Compute pairwise distances
    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        normed = matrix / norms
        cos_sims = normed @ normed.T
        # Cosine distance = 1 - cosine_similarity
        dist_matrix = 1.0 - cos_sims
    else:
        dist_matrix = cdist(matrix, matrix, metric="euclidean").astype(np.float32)

    # Classify all pairs by hierarchy level
    families = [id_to_meta[pid]["family"] for pid in pids]
    superfamilies = [id_to_meta[pid]["superfamily"] for pid in pids]
    folds = [id_to_meta[pid]["fold"] for pid in pids]

    within_family = []
    same_superfamily = []
    same_fold = []
    unrelated = []

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if families[i] == families[j]:
                within_family.append(d)
            elif superfamilies[i] == superfamilies[j]:
                same_superfamily.append(d)
            elif folds[i] == folds[j]:
                same_fold.append(d)
            else:
                unrelated.append(d)

    # Subsample large levels for speed
    def _subsample(arr, max_n):
        if len(arr) <= max_n:
            return np.array(arr, dtype=np.float32)
        idx = rng.choice(len(arr), max_n, replace=False)
        return np.array(arr, dtype=np.float32)[idx]

    within_family = _subsample(within_family, max_pairs_per_level)
    same_superfamily = _subsample(same_superfamily, max_pairs_per_level)
    same_fold = _subsample(same_fold, max_pairs_per_level)
    unrelated = _subsample(unrelated, max_pairs_per_level)

    levels = {
        "within_family": within_family,
        "same_superfamily": same_superfamily,
        "same_fold": same_fold,
        "unrelated": unrelated,
    }

    results = {"metric": metric, "n_proteins": n}

    for level_name, dists in levels.items():
        if len(dists) > 0:
            results[f"{level_name}_mean"] = float(np.mean(dists))
            results[f"{level_name}_std"] = float(np.std(dists))
            results[f"{level_name}_median"] = float(np.median(dists))
            results[f"{level_name}_n_pairs"] = len(dists)
        else:
            results[f"{level_name}_mean"] = None
            results[f"{level_name}_n_pairs"] = 0

    # Separation ratio: unrelated / within_family (higher = better)
    wf_mean = results.get("within_family_mean")
    ur_mean = results.get("unrelated_mean")
    if wf_mean and wf_mean > 1e-8 and ur_mean:
        results["separation_ratio"] = float(ur_mean / wf_mean)
    else:
        results["separation_ratio"] = None

    # Check ordering: within_family < same_SF < same_fold < unrelated
    means = []
    for level_name in ["within_family", "same_superfamily", "same_fold", "unrelated"]:
        m = results.get(f"{level_name}_mean")
        if m is not None:
            means.append(m)
    results["ordering_correct"] = all(means[i] < means[i + 1] for i in range(len(means) - 1))

    return results


def plot_distance_distributions(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    metric: str = "cosine",
    output_path: str = "distance_distributions.png",
    title: str | None = None,
    max_pairs_per_level: int = 50_000,
    seed: int = 42,
) -> dict:
    """Plot histograms of within-family vs between-family distances.

    Args:
        vectors: {protein_id: embedding_vector}.
        metadata: List of dicts with id, family, superfamily, fold.
        metric: "cosine" or "euclidean".
        output_path: Path to save PNG.
        title: Plot title (auto-generated if None).
        max_pairs_per_level: Cap on pairs sampled per level.
        seed: Random seed.

    Returns:
        Dict with hierarchy distances (same as evaluate_hierarchy_distances).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)

    # Build lookup
    id_to_meta = {}
    for m in metadata:
        pid = m["id"]
        if pid in vectors and all(k in m for k in ("family", "superfamily", "fold")):
            id_to_meta[pid] = m

    pids = list(id_to_meta.keys())
    n = len(pids)
    matrix = np.array([vectors[pid] for pid in pids], dtype=np.float32)

    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        normed = matrix / norms
        dist_matrix = 1.0 - (normed @ normed.T)
    else:
        dist_matrix = cdist(matrix, matrix, metric="euclidean").astype(np.float32)

    families = [id_to_meta[pid]["family"] for pid in pids]

    within = []
    between = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if families[i] == families[j]:
                within.append(d)
            else:
                between.append(d)

    # Subsample
    def _sub(arr, max_n):
        if len(arr) <= max_n:
            return np.array(arr, dtype=np.float32)
        idx = rng.choice(len(arr), max_n, replace=False)
        return np.array(arr, dtype=np.float32)[idx]

    within = _sub(within, max_pairs_per_level)
    between = _sub(between, max_pairs_per_level)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.linspace(
        min(within.min(), between.min()),
        max(within.max(), between.max()),
        80,
    )

    ax.hist(within, bins=bins, alpha=0.6, label=f"Within-family (n={len(within):,})",
            color="#2196F3", density=True)
    ax.hist(between, bins=bins, alpha=0.6, label=f"Between-family (n={len(between):,})",
            color="#F44336", density=True)

    ax.axvline(np.mean(within), color="#1565C0", linestyle="--", linewidth=2,
               label=f"Within mean={np.mean(within):.3f}")
    ax.axvline(np.mean(between), color="#C62828", linestyle="--", linewidth=2,
               label=f"Between mean={np.mean(between):.3f}")

    dist_label = "Cosine Distance" if metric == "cosine" else "Euclidean Distance"
    ax.set_xlabel(dist_label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    if title is None:
        title = f"Distance Distributions ({metric.title()})"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    # Compute overlap coefficient (area of intersection)
    h_within, _ = np.histogram(within, bins=bins, density=True)
    h_between, _ = np.histogram(between, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = float(np.sum(np.minimum(h_within, h_between)) * bin_width)

    return {
        "within_mean": float(np.mean(within)),
        "between_mean": float(np.mean(between)),
        "within_std": float(np.std(within)),
        "between_std": float(np.std(between)),
        "overlap_coefficient": overlap,
        "n_within": len(within),
        "n_between": len(between),
        "plot_path": output_path,
    }
