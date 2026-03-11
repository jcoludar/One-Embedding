"""Embedding quality metrics from the literature.

Implements:
  - Random Neighbor Score (RNS) from Prabakaran & Bromberg (2025)
  - Inherent information metrics from Senoner, Koludarov et al. (2025)
"""

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans


def compute_rns(
    vectors: dict[str, np.ndarray],
    n_random: int = 1000,
    k: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Random Neighbor Score (Prabakaran & Bromberg 2025).

    Generates n_random random vectors in the same space, mixes them with
    real vectors, and for each real protein computes what fraction of its
    k nearest neighbors are random.

    RNS ~ 0.0: embeddings are well-separated from random noise
    RNS > 0.6: embeddings are unreliable (indistinguishable from random)

    Returns:
        rns_mean: Mean RNS across all real proteins.
        rns_std: Std of RNS across all real proteins.
        rns_median: Median RNS.
        n_unreliable: Number of proteins with RNS > 0.6.
        frac_unreliable: Fraction of proteins with RNS > 0.6.
    """
    rng = np.random.RandomState(seed)

    pids = list(vectors.keys())
    n_real = len(pids)
    if n_real < 2:
        return {"rns_mean": float("nan"), "rns_std": float("nan"),
                "rns_median": float("nan"), "n_unreliable": 0,
                "frac_unreliable": float("nan")}

    dim = vectors[pids[0]].shape[0]

    # Build real matrix
    real_matrix = np.array([vectors[pid] for pid in pids])

    # Generate random vectors matching the distribution of real vectors
    # Use same mean and std per dimension to make a fair test
    col_mean = real_matrix.mean(axis=0)
    col_std = real_matrix.std(axis=0) + 1e-8
    random_matrix = rng.randn(n_random, dim) * col_std + col_mean

    # Combined matrix: real first, then random
    combined = np.vstack([real_matrix, random_matrix])
    is_random = np.array([False] * n_real + [True] * n_random)

    # Normalize for cosine similarity
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms[norms == 0] = 1
    combined_normed = combined / norms

    # Compute similarities for real proteins against all
    # (n_real, n_real + n_random)
    sims = real_matrix / np.linalg.norm(real_matrix, axis=1, keepdims=True).clip(1e-8) \
        @ combined_normed.T

    rns_scores = []
    for i in range(n_real):
        row = sims[i].copy()
        row[i] = -np.inf  # exclude self
        top_k_idx = np.argsort(row)[-k:]
        n_random_in_topk = is_random[top_k_idx].sum()
        rns_scores.append(n_random_in_topk / k)

    rns_arr = np.array(rns_scores)
    threshold = 0.6

    return {
        "rns_mean": float(rns_arr.mean()),
        "rns_std": float(rns_arr.std()),
        "rns_median": float(np.median(rns_arr)),
        "n_unreliable": int((rns_arr > threshold).sum()),
        "frac_unreliable": float((rns_arr > threshold).mean()),
        "n_proteins": n_real,
        "n_random": n_random,
        "k": k,
    }


def compute_inherent_information(
    vectors: dict[str, np.ndarray],
    metadata: list[dict],
    label_key: str = "family",
) -> dict[str, float]:
    """Measure inherent clustering quality without any training.

    Computes metrics on raw embedding distances (no learned transformation).
    Based on the inherent/extractable framework from Senoner, Koludarov et al. (2025).

    Metrics:
      - silhouette_score: How well clusters match labels (-1 to 1, higher = better)
      - knn_purity_k1: Fraction of proteins whose nearest neighbor shares the same label
      - knn_purity_k5: Fraction of k=5 nearest neighbors that share the same label
      - ami: Adjusted Mutual Information between KMeans clusters and true labels

    Args:
        vectors: {protein_id: vector} dict of pooled embeddings.
        metadata: List of dicts with 'id' and label columns.
        label_key: Which label to evaluate against.

    Returns dict with all metrics.
    """
    id_to_label = {m["id"]: m[label_key] for m in metadata if label_key in m}
    pids = [pid for pid in vectors if pid in id_to_label]

    if len(pids) < 10:
        return {"silhouette": float("nan"), "knn_purity_k1": float("nan"),
                "knn_purity_k5": float("nan"), "ami": float("nan")}

    X = np.array([vectors[pid] for pid in pids])
    labels = [id_to_label[pid] for pid in pids]

    # Encode labels to integers
    unique_labels = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels])
    n_classes = len(unique_labels)

    # Normalize for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_normed = X / norms

    # Silhouette score (cosine metric)
    sil = float("nan")
    if n_classes >= 2 and n_classes < len(pids):
        try:
            sil = float(silhouette_score(X_normed, y, metric="cosine"))
        except ValueError:
            pass

    # kNN purity
    sims = X_normed @ X_normed.T
    np.fill_diagonal(sims, -np.inf)

    purities_k1 = []
    purities_k5 = []
    for i in range(len(pids)):
        sorted_idx = np.argsort(sims[i])[::-1]
        # k=1
        purities_k1.append(float(y[sorted_idx[0]] == y[i]))
        # k=5
        top5 = sorted_idx[:5]
        purities_k5.append(float(np.mean(y[top5] == y[i])))

    # AMI: KMeans with n_classes clusters vs true labels
    ami = float("nan")
    if n_classes >= 2:
        try:
            kmeans = KMeans(n_clusters=min(n_classes, len(pids) - 1),
                            random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(X_normed)
            ami = float(adjusted_mutual_info_score(y, pred_labels))
        except Exception:
            pass

    return {
        "silhouette": sil,
        "knn_purity_k1": float(np.mean(purities_k1)),
        "knn_purity_k5": float(np.mean(purities_k5)),
        "ami": ami,
        "n_classes": n_classes,
        "n_proteins": len(pids),
    }


def compute_token_diversity(
    model,
    embeddings: dict[str, np.ndarray],
    device=None,
    max_len: int = 512,
) -> dict[str, float]:
    """Measure diversity/redundancy of K latent tokens across proteins.

    For each protein, compresses to K tokens and measures:
    - Mean pairwise cosine similarity between tokens (1.0 = identical)
    - Effective rank of K×D' matrix (1.0 = all tokens identical, K = maximally diverse)

    Args:
        model: AttentionPoolCompressor (must have compress method)
        embeddings: Per-residue embeddings {id: (L, D)}
        device: torch device
        max_len: Max sequence length to process

    Returns:
        Dict with mean_pairwise_cos, std_pairwise_cos, mean_effective_rank, std_effective_rank
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        from src.utils.device import get_device
        device = get_device()

    model = model.to(device)
    model.eval()

    pairwise_cosines = []
    effective_ranks = []

    with torch.no_grad():
        for pid, emb in embeddings.items():
            L = min(emb.shape[0], max_len)
            states = torch.from_numpy(emb[:L]).unsqueeze(0).float().to(device)
            mask = torch.ones(1, L, device=device)
            latent = model.compress(states, mask)  # (1, K, D')

            tokens = latent[0]  # (K, D')
            K = tokens.shape[0]

            # Pairwise cosine similarity
            normed = F.normalize(tokens, dim=-1)
            sim_matrix = normed @ normed.T  # (K, K)
            # Extract upper triangle (exclude diagonal)
            mask_ut = torch.triu(torch.ones(K, K, device=device), diagonal=1).bool()
            pairwise_cos = sim_matrix[mask_ut].mean().item()
            pairwise_cosines.append(pairwise_cos)

            # Effective rank via singular values
            # effective_rank = exp(entropy of normalized singular values)
            sv = torch.linalg.svdvals(tokens.cpu())  # (min(K, D'),)
            sv_norm = sv / sv.sum()
            sv_norm = sv_norm[sv_norm > 1e-10]  # avoid log(0)
            entropy = -(sv_norm * sv_norm.log()).sum().item()
            eff_rank = np.exp(entropy)
            effective_ranks.append(eff_rank)

    return {
        "mean_pairwise_cos": float(np.mean(pairwise_cosines)),
        "std_pairwise_cos": float(np.std(pairwise_cosines)),
        "mean_effective_rank": float(np.mean(effective_ranks)),
        "std_effective_rank": float(np.std(effective_ranks)),
        "n_proteins": len(pairwise_cosines),
    }
