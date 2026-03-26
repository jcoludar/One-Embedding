"""Bootstrap CI and multi-seed statistics — mandatory for all metrics.

Every metric in the rigorous benchmark must go through bootstrap_ci() to get
a MetricResult with confidence intervals. No bare floats allowed (Rule 4).

Uses BCa (bias-corrected and accelerated) bootstrap for second-order accurate
CIs (DiCiccio & Efron 1996). Falls back to percentile for n<25 where
jackknife is unstable, or if BCa raises an exception.
"""

from typing import Callable

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rules import MetricResult

# Minimum sample size for BCa (jackknife is unstable below this)
_BCA_MIN_N = 25


def bootstrap_ci(
    scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    """Compute a metric with bootstrap 95% CI (Rule 4).

    Uses BCa (bias-corrected and accelerated) method for n>=25, falling back
    to percentile for small samples or if BCa raises an exception.

    Args:
        scores: {item_id: score} — per-query, per-protein, or per-residue.
        metric_fn: Aggregation function (default: np.mean).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        MetricResult with value, ci_lower, ci_upper, n, ci_method.
    """
    ids = sorted(scores.keys())
    values = np.array([scores[k] for k in ids], dtype=np.float64)
    n = len(values)

    observed = float(metric_fn(values))

    # Choose method based on sample size
    use_bca = n >= _BCA_MIN_N
    method = "BCa" if use_bca else "percentile"

    def statistic(x, axis):
        return np.apply_along_axis(metric_fn, axis, x)

    try:
        result = scipy_bootstrap(
            (values,),
            statistic=statistic,
            n_resamples=n_bootstrap,
            method=method,
            confidence_level=1 - alpha,
            random_state=np.random.RandomState(seed),
        )
        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
        ci_method = "bca" if use_bca else "percentile"

        # NaN fallback check
        if np.isnan(ci_lower) or np.isnan(ci_upper):
            raise ValueError("BCa returned NaN")
    except Exception:
        # Fallback to percentile if BCa fails
        result = scipy_bootstrap(
            (values,),
            statistic=statistic,
            n_resamples=n_bootstrap,
            method="percentile",
            confidence_level=1 - alpha,
            random_state=np.random.RandomState(seed),
        )
        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
        ci_method = "percentile"

    return MetricResult(
        value=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=n,
        ci_method=ci_method,
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

    Uses paired BCa resampling — the same items are drawn for both raw and
    compressed in each bootstrap iteration, so correlated noise cancels.
    Falls back to percentile for n<25 or if BCa raises an exception.

    Args:
        raw_scores: {item_id: score} for the raw (uncompressed) system.
        comp_scores: {item_id: score} for the compressed system.
        metric_fn: Aggregation (default: np.mean).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.
        alpha: Significance level.

    Returns:
        MetricResult where value is retention percentage (0-100+),
        ci_lower/ci_upper are the BCa bootstrap bounds, ci_method is set.
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

    use_bca = n >= _BCA_MIN_N
    method = "BCa" if use_bca else "percentile"

    def retention_statistic(raw, comp, axis):
        raw_agg = np.apply_along_axis(metric_fn, axis, raw)
        comp_agg = np.apply_along_axis(metric_fn, axis, comp)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = np.where(raw_agg != 0, comp_agg / raw_agg * 100, 0.0)
        return ret

    try:
        result = scipy_bootstrap(
            (raw_arr, comp_arr),
            statistic=retention_statistic,
            paired=True,
            n_resamples=n_bootstrap,
            method=method,
            confidence_level=1 - alpha,
            random_state=np.random.RandomState(seed),
        )
        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
        ci_method = "bca" if use_bca else "percentile"

        # NaN fallback check
        if np.isnan(ci_lower) or np.isnan(ci_upper):
            raise ValueError("BCa returned NaN")
    except Exception:
        # Fallback to percentile
        result = scipy_bootstrap(
            (raw_arr, comp_arr),
            statistic=retention_statistic,
            paired=True,
            n_resamples=n_bootstrap,
            method="percentile",
            confidence_level=1 - alpha,
            random_state=np.random.RandomState(seed),
        )
        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
        ci_method = "percentile"

    return MetricResult(
        value=retention_obs,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=n,
        ci_method=ci_method,
    )


def paired_bootstrap_metric(
    raw_scores: dict[str, float],
    comp_scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[MetricResult, MetricResult]:
    """Bootstrap CIs on two systems evaluated on the same items.

    Computes independent BCa bootstrap CIs for raw and compressed, using the
    same seed for reproducibility. Each system gets its own CI (not paired
    difference CIs). Use for pooled metrics (e.g., Spearman rho on pooled
    residues) where you need CIs on both raw and compressed.

    Falls back to percentile for n<25 or if BCa raises an exception.

    Returns:
        (raw_metric, comp_metric) — both MetricResult with independent CIs.
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

    use_bca = n >= _BCA_MIN_N
    method = "BCa" if use_bca else "percentile"

    def statistic(x, axis):
        return np.apply_along_axis(metric_fn, axis, x)

    def _compute_ci(arr, obs, rng_seed):
        """Compute BCa CI for a single array, with percentile fallback."""
        try:
            result = scipy_bootstrap(
                (arr,),
                statistic=statistic,
                n_resamples=n_bootstrap,
                method=method,
                confidence_level=1 - alpha,
                random_state=np.random.RandomState(rng_seed),
            )
            ci_lo = float(result.confidence_interval.low)
            ci_hi = float(result.confidence_interval.high)
            ci_m = "bca" if use_bca else "percentile"

            if np.isnan(ci_lo) or np.isnan(ci_hi):
                raise ValueError("BCa returned NaN")
        except Exception:
            result = scipy_bootstrap(
                (arr,),
                statistic=statistic,
                n_resamples=n_bootstrap,
                method="percentile",
                confidence_level=1 - alpha,
                random_state=np.random.RandomState(rng_seed),
            )
            ci_lo = float(result.confidence_interval.low)
            ci_hi = float(result.confidence_interval.high)
            ci_m = "percentile"
        return ci_lo, ci_hi, ci_m

    raw_lo, raw_hi, raw_ci_m = _compute_ci(raw_arr, raw_obs, seed)
    comp_lo, comp_hi, comp_ci_m = _compute_ci(comp_arr, comp_obs, seed)

    raw_result = MetricResult(
        value=raw_obs,
        ci_lower=raw_lo,
        ci_upper=raw_hi,
        n=n,
        ci_method=raw_ci_m,
    )
    comp_result = MetricResult(
        value=comp_obs,
        ci_lower=comp_lo,
        ci_upper=comp_hi,
        n=n,
        ci_method=comp_ci_m,
    )
    return raw_result, comp_result


def multi_seed_summary(seed_results: list[MetricResult]) -> MetricResult:
    """Aggregate results across multiple seeds (Rule 5) -- DEPRECATED.

    Prefer averaged_multi_seed() which averages per-item predictions
    across seeds before bootstrapping, properly capturing both probe
    training variance and sampling variance.

    This function selects the median-performing seed's CI, which ignores
    probe training variability (Bouthillier et al. 2021).

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


def averaged_multi_seed(
    seed_scores: list[dict[str, float]],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    """Average per-item scores across seeds, then bootstrap (Rule 5).

    Instead of picking the median seed's CI (which ignores probe variance),
    this averages predictions across seeds for each item, then computes
    a single BCa bootstrap CI on the averaged scores.

    Per-seed aggregate values are stored in seeds_mean/seeds_std for
    transparency about probe training variability.

    Args:
        seed_scores: List of {item_id: score} dicts, one per seed.
        metric_fn: Aggregation function (default: np.mean).
        n_bootstrap: Bootstrap iterations.
        seed: Random seed for bootstrap.
        alpha: Significance level.

    Returns:
        MetricResult with CI from bootstrapping the averaged scores,
        plus seeds_mean/seeds_std from per-seed aggregates.
    """
    if len(seed_scores) == 0:
        raise ValueError("Need at least one seed's scores.")

    # Find common items across all seeds
    common = sorted(set.intersection(*[set(s.keys()) for s in seed_scores]))
    if len(common) == 0:
        raise ValueError("No common items across seeds.")

    # Average per-item scores across seeds
    averaged = {}
    for item_id in common:
        vals = [s[item_id] for s in seed_scores]
        averaged[item_id] = float(np.mean(vals))

    # Per-seed aggregate values (for transparency)
    per_seed_aggregates = []
    for s in seed_scores:
        vals = np.array([s[k] for k in common], dtype=np.float64)
        per_seed_aggregates.append(float(metric_fn(vals)))

    seeds_mean = float(np.mean(per_seed_aggregates))
    seeds_std = float(np.std(per_seed_aggregates, ddof=1)) if len(per_seed_aggregates) > 1 else 0.0

    # Bootstrap CI on the averaged scores
    result = bootstrap_ci(averaged, metric_fn=metric_fn, n_bootstrap=n_bootstrap,
                          seed=seed, alpha=alpha)

    return MetricResult(
        value=result.value,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        n=result.n,
        seeds_mean=seeds_mean,
        seeds_std=seeds_std,
        ci_method=result.ci_method,
    )
