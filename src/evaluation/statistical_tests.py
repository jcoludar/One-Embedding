"""Statistical significance tests for embedding evaluation.

Provides paired bootstrap, permutation tests, and effect sizes for comparing
retrieval/classification metrics between embedding configurations.
"""

import numpy as np


def paired_bootstrap_test(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for per-query score comparison.

    Tests H0: mean(scores_a) == mean(scores_b) using paired resampling.

    Args:
        scores_a: {query_id: score} for system A.
        scores_b: {query_id: score} for system B (same query IDs).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        dict with: mean_a, mean_b, mean_diff, ci_lower, ci_upper, p_value.
    """
    rng = np.random.RandomState(seed)

    # Align query IDs
    common_ids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if len(common_ids) == 0:
        raise ValueError("No common query IDs between scores_a and scores_b")

    a = np.array([scores_a[qid] for qid in common_ids])
    b = np.array([scores_b[qid] for qid in common_ids])
    diffs = a - b
    observed_diff = diffs.mean()

    # Bootstrap the difference
    n = len(diffs)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Two-sided p-value: proportion of bootstrap diffs on the other side of 0
    if observed_diff >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(observed_diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_queries": len(common_ids),
        "n_bootstrap": n_bootstrap,
    }


def multi_seed_permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation test for comparing n-seed metric distributions.

    For small n (e.g., 3 seeds), uses exhaustive permutation when feasible.

    Args:
        scores_a: List of metric values for system A (one per seed).
        scores_b: List of metric values for system B (one per seed).
        n_permutations: Number of random permutations (used if exhaustive is too large).
        seed: Random seed.

    Returns:
        dict with: mean_a, mean_b, mean_diff, p_value.
    """
    rng = np.random.RandomState(seed)

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    observed_diff = a.mean() - b.mean()

    # Pool all scores
    pooled = np.concatenate([a, b])
    n_a = len(a)
    n_total = len(pooled)

    # Exhaustive if feasible (2^n_total permutations)
    from math import comb
    total_perms = comb(n_total, n_a)

    if total_perms <= 10000:
        # Exhaustive
        from itertools import combinations
        count = 0
        for idx in combinations(range(n_total), n_a):
            perm_a = pooled[list(idx)]
            perm_diff = perm_a.mean() - (pooled.sum() - perm_a.sum()) / (n_total - n_a)
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        p_value = count / total_perms
    else:
        # Random permutation
        count = 0
        for _ in range(n_permutations):
            rng.shuffle(pooled)
            perm_diff = pooled[:n_a].mean() - pooled[n_a:].mean()
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        p_value = (count + 1) / (n_permutations + 1)

    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_a": len(a),
        "n_b": len(b),
        "exhaustive": total_perms <= 10000,
    }


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns positive d when a > b.
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)

    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return float((a.mean() - b.mean()) / pooled_std)
