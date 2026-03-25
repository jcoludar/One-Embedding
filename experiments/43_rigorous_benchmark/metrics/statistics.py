"""Bootstrap CI and multi-seed statistics — mandatory for all metrics.

Every metric in the rigorous benchmark must go through bootstrap_ci() to get
a MetricResult with confidence intervals. No bare floats allowed (Rule 4).
"""

from typing import Callable

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rules import MetricResult


def bootstrap_ci(
    scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    """Compute a metric with bootstrap 95% CI (Rule 4).

    Args:
        scores: {item_id: score} — per-query, per-protein, or per-residue.
        metric_fn: Aggregation function (default: np.mean).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        MetricResult with value, ci_lower, ci_upper, n.
    """
    ids = sorted(scores.keys())
    values = np.array([scores[k] for k in ids], dtype=np.float64)
    n = len(values)

    observed = float(metric_fn(values))

    rng = np.random.RandomState(seed)
    # Vectorized bootstrap for speed
    idx_matrix = rng.randint(0, n, size=(n_bootstrap, n))
    boot_samples = values[idx_matrix]  # (n_bootstrap, n)
    boot_values = np.apply_along_axis(metric_fn, 1, boot_samples)

    ci_lower = float(np.percentile(boot_values, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))

    return MetricResult(
        value=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=n,
    )


def paired_bootstrap_retention(
    raw_scores: dict[str, float],
    comp_scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    """Bootstrap CI on the retention ratio: metric(compressed) / metric(raw).

    Uses paired resampling — the same items are drawn for both raw and
    compressed in each bootstrap iteration, so correlated noise cancels.

    Args:
        raw_scores: {item_id: score} for the raw (uncompressed) system.
        comp_scores: {item_id: score} for the compressed system.
        metric_fn: Aggregation (default: np.mean).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.
        alpha: Significance level.

    Returns:
        MetricResult where value is retention percentage (0-100+),
        ci_lower/ci_upper are the bootstrap percentile bounds.
    """
    # Align on common IDs
    common = sorted(set(raw_scores.keys()) & set(comp_scores.keys()))
    if len(common) == 0:
        return MetricResult(value=0.0, ci_lower=0.0, ci_upper=0.0, n=0)

    raw_arr = np.array([raw_scores[k] for k in common], dtype=np.float64)
    comp_arr = np.array([comp_scores[k] for k in common], dtype=np.float64)
    n = len(common)

    raw_obs = float(metric_fn(raw_arr))
    comp_obs = float(metric_fn(comp_arr))
    retention_obs = (comp_obs / raw_obs * 100) if raw_obs != 0 else 0.0

    rng = np.random.RandomState(seed)
    idx_matrix = rng.randint(0, n, size=(n_bootstrap, n))

    raw_boot = np.apply_along_axis(metric_fn, 1, raw_arr[idx_matrix])
    comp_boot = np.apply_along_axis(metric_fn, 1, comp_arr[idx_matrix])

    # Avoid division by zero in bootstrap samples
    valid = raw_boot != 0
    retention_boot = np.full(n_bootstrap, np.nan)
    retention_boot[valid] = (comp_boot[valid] / raw_boot[valid]) * 100

    # Drop NaN (from zero-denominator samples)
    retention_boot = retention_boot[~np.isnan(retention_boot)]

    ci_lower = float(np.percentile(retention_boot, 100 * alpha / 2))
    ci_upper = float(np.percentile(retention_boot, 100 * (1 - alpha / 2)))

    return MetricResult(
        value=retention_obs,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=n,
    )


def paired_bootstrap_metric(
    raw_scores: dict[str, float],
    comp_scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[MetricResult, MetricResult]:
    """Bootstrap CIs on two paired systems simultaneously.

    Same resampling for both — gives correlated CIs suitable for
    paired comparison. Use for pooled metrics (e.g., Spearman rho on
    pooled residues) where you need CIs on both raw and compressed.

    Returns:
        (raw_metric, comp_metric) — both MetricResult with paired CIs.
    """
    common = sorted(set(raw_scores.keys()) & set(comp_scores.keys()))
    if len(common) == 0:
        empty = MetricResult(value=0.0, ci_lower=0.0, ci_upper=0.0, n=0)
        return empty, empty

    raw_arr = np.array([raw_scores[k] for k in common], dtype=np.float64)
    comp_arr = np.array([comp_scores[k] for k in common], dtype=np.float64)
    n = len(common)

    raw_obs = float(metric_fn(raw_arr))
    comp_obs = float(metric_fn(comp_arr))

    rng = np.random.RandomState(seed)
    idx_matrix = rng.randint(0, n, size=(n_bootstrap, n))

    raw_boot = np.apply_along_axis(metric_fn, 1, raw_arr[idx_matrix])
    comp_boot = np.apply_along_axis(metric_fn, 1, comp_arr[idx_matrix])

    raw_result = MetricResult(
        value=raw_obs,
        ci_lower=float(np.percentile(raw_boot, 100 * alpha / 2)),
        ci_upper=float(np.percentile(raw_boot, 100 * (1 - alpha / 2))),
        n=n,
    )
    comp_result = MetricResult(
        value=comp_obs,
        ci_lower=float(np.percentile(comp_boot, 100 * alpha / 2)),
        ci_upper=float(np.percentile(comp_boot, 100 * (1 - alpha / 2))),
        n=n,
    )
    return raw_result, comp_result


def multi_seed_summary(seed_results: list[MetricResult]) -> MetricResult:
    """Aggregate results across multiple seeds (Rule 5).

    Selects the median-performing seed for the headline number and CI.
    Reports mean +/- std across all seeds.

    Args:
        seed_results: List of MetricResult, one per seed.

    Returns:
        MetricResult from the median seed, with seeds_mean and seeds_std populated.
    """
    values = [r.value for r in seed_results]
    seeds_mean = float(np.mean(values))
    seeds_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Select median seed
    sorted_idx = np.argsort(values)
    median_idx = sorted_idx[len(sorted_idx) // 2]
    median_result = seed_results[median_idx]

    return MetricResult(
        value=median_result.value,
        ci_lower=median_result.ci_lower,
        ci_upper=median_result.ci_upper,
        n=median_result.n,
        seeds_mean=seeds_mean,
        seeds_std=seeds_std,
    )
