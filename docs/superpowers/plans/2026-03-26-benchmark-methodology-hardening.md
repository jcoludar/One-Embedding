# Benchmark Methodology Hardening — Nature-Level Rigor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Exp 43 benchmark framework from "good" to "Nature-publication-level" by fixing statistical methodology, correcting disorder evaluation to match published standards, and adding a novel ABTT cross-corpus stability test.

**Architecture:** Three independent code changes to the benchmark framework (statistics, disorder runner, ABTT stability test), each with full TDD, followed by a benchmark re-run phase. All changes are in `experiments/43_rigorous_benchmark/` and `tests/`.

**Tech Stack:** numpy, scipy 1.17.1 (`scipy.stats.bootstrap` available since 1.7.0, BCa supported), scikit-learn, pytest

**Key Citations:**
- Efron & Tibshirani (1993) *An Introduction to the Bootstrap* — BCa
- DiCiccio & Efron (1996) "Bootstrap Confidence Intervals" — second-order accuracy
- Nielsen & Mulder (2019) *Scientific Reports* — CheZOD evaluation: pooled Spearman ρ
- CAID Round 3 (2025) — Fmax, AUC, pooled residue-level evaluation
- Davison & Hinkley (1997) — cluster bootstrap, B=10,000 sufficient
- Bouthillier et al. (2021) *MLSys* — seed variance in benchmarks

---

## File Structure

### Files to Modify
| File | Responsibility | Changes |
|------|---------------|---------|
| `experiments/43_rigorous_benchmark/metrics/statistics.py` | Bootstrap CI, multi-seed aggregation | BCa bootstrap, averaged multi-seed, cluster bootstrap for pooled metrics |
| `experiments/43_rigorous_benchmark/runners/per_residue.py` | SS3, SS8, disorder benchmark runners | Averaged multi-seed for SS3/SS8, pooled ρ + AUC-ROC + cluster bootstrap for disorder |
| `experiments/43_rigorous_benchmark/rules.py` | MetricResult dataclass, golden rules | Add `ci_method` field to MetricResult |
| `tests/test_benchmark_statistics.py` | Tests for statistics module | New tests for BCa, averaged seed, cluster bootstrap |
| `tests/test_benchmark_rules.py` | Tests for golden rules | Test for ci_method field |
| `tests/test_benchmark_per_residue_runner.py` | Runner integration tests | Update disorder return dict key from `"spearman_rho"` to `"pooled_spearman_rho"` |
| `experiments/43_rigorous_benchmark/run_phase_a1.py` | Phase A benchmark script | Update `pooled_spearman_rho` access from bare float to `.value` (3 sites) |
| `experiments/43_rigorous_benchmark/run_phase_b.py` | Phase B benchmark script | Update `pooled_spearman_rho` access from bare float to `.value` (4 sites) |

### Files to Create
| File | Responsibility |
|------|---------------|
| `experiments/43_rigorous_benchmark/metrics/abtt_stability.py` | ABTT cross-corpus stability test (principal angles, downstream performance) |
| `tests/test_benchmark_disorder.py` | Dedicated disorder evaluation tests (pooled ρ, AUC-ROC, cluster bootstrap) |
| `tests/test_abtt_stability.py` | Tests for cross-corpus stability functions |

### Files to Read (reference only)
| File | Why |
|------|-----|
| `src/one_embedding/core/preprocessing.py` | ABTT fit_abtt / apply_abtt implementation |
| `src/one_embedding/preprocessing.py` | compute_corpus_stats / all_but_the_top implementation |
| `experiments/43_rigorous_benchmark/config.py` | Paths, seeds, thresholds |
| `experiments/43_rigorous_benchmark/probes/linear.py` | CV-tuned probes (no changes needed) |
| `experiments/43_rigorous_benchmark/runners/protein_level.py` | Retrieval runner (needs BCa update) |

---

## Task 1: BCa Bootstrap with Percentile Fallback

**Files:**
- Modify: `experiments/43_rigorous_benchmark/rules.py:17-33`
- Modify: `experiments/43_rigorous_benchmark/metrics/statistics.py:17-56`
- Test: `tests/test_benchmark_statistics.py`

### Rationale
The percentile bootstrap is first-order accurate (coverage error O(1/√n)). BCa (bias-corrected and accelerated) is second-order accurate (O(1/n)). For n=103, this is the difference between ~10% and ~1% coverage error. scipy 1.17.1 provides `scipy.stats.bootstrap(method='BCa')`. BCa requires a jackknife acceleration estimate that can be unstable for n<25 — we fall back to percentile there.

- [ ] **Step 1: Add ci_method to MetricResult**

In `experiments/43_rigorous_benchmark/rules.py`, add an optional `ci_method` field:

```python
@dataclass
class MetricResult:
    """Every metric MUST be wrapped in this. No bare floats allowed."""
    value: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    n: int
    seeds_mean: Optional[float] = None
    seeds_std: Optional[float] = None
    ci_method: str = "percentile"  # "bca" or "percentile"

    def __post_init__(self):
        if self.ci_lower is None or self.ci_upper is None:
            raise ValueError(
                "Rule 4 violation: MetricResult requires ci_lower and ci_upper. "
                "Use bootstrap_ci() to compute confidence intervals."
            )
```

- [ ] **Step 2: Write failing tests for BCa bootstrap**

In `tests/test_benchmark_statistics.py`, add:

```python
class TestBCaBootstrap:

    def test_bca_returns_metric_result(self):
        scores = {f"q{i}": float(i % 2) for i in range(200)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=2000, seed=42)
        assert isinstance(result, MetricResult)
        assert result.ci_method == "bca"

    def test_bca_ci_contains_observed(self):
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.rand()) for i in range(500)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=5000, seed=42)
        assert result.ci_lower <= result.value <= result.ci_upper

    def test_bca_asymmetric_for_skewed_data(self):
        """BCa should produce asymmetric CIs for skewed data (upper > lower)."""
        # Heavily right-skewed data (exponential-like)
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.exponential(1.0)) for i in range(200)}
        result_bca = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=9999, seed=42)
        assert result_bca.ci_method == "bca"
        # BCa CI should be strictly asymmetric for right-skewed data:
        # upper tail wider than lower tail
        mid = result_bca.value
        lower_dist = mid - result_bca.ci_lower
        upper_dist = result_bca.ci_upper - mid
        assert upper_dist > lower_dist, (
            f"BCa should produce asymmetric CI for right-skewed data: "
            f"upper_dist={upper_dist:.4f} should be > lower_dist={lower_dist:.4f}"
        )

    def test_fallback_to_percentile_for_small_n(self):
        """With n < 25, BCa jackknife can be unstable — should fall back."""
        scores = {f"q{i}": float(i) for i in range(10)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=2000, seed=42)
        assert result.ci_method == "percentile"

    def test_bca_deterministic_with_seed(self):
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.rand()) for i in range(100)}
        r1 = bootstrap_ci(scores, n_bootstrap=5000, seed=42)
        r2 = bootstrap_ci(scores, n_bootstrap=5000, seed=42)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_statistics.py::TestBCaBootstrap -v`
Expected: FAIL — `ci_method` attribute doesn't exist, bootstrap_ci still uses percentile

- [ ] **Step 4: Implement BCa bootstrap**

Replace `bootstrap_ci` in `experiments/43_rigorous_benchmark/metrics/statistics.py`:

```python
def bootstrap_ci(
    scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    """Compute a metric with bootstrap 95% CI using BCa method (Rule 4).

    Uses BCa (bias-corrected and accelerated) bootstrap for second-order
    accurate CIs (DiCiccio & Efron, 1996). Falls back to percentile for
    n < 25 where the jackknife acceleration estimate is unstable.

    Args:
        scores: {item_id: score} — per-query, per-protein, or per-residue.
        metric_fn: Aggregation function (default: np.mean).
        n_bootstrap: Number of bootstrap iterations (default 10,000).
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        MetricResult with value, ci_lower, ci_upper, n, ci_method.
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    ids = sorted(scores.keys())
    values = np.array([scores[k] for k in ids], dtype=np.float64)
    n = len(values)

    observed = float(metric_fn(values))

    # BCa requires jackknife — unstable for very small n
    use_bca = n >= 25

    try:
        result = scipy_bootstrap(
            (values,),
            statistic=lambda x, axis: np.apply_along_axis(metric_fn, axis, x),
            n_resamples=n_bootstrap,
            confidence_level=1.0 - alpha,
            method="BCa" if use_bca else "percentile",
            random_state=np.random.RandomState(seed),
        )
        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
        ci_method = "bca" if use_bca else "percentile"
    except Exception:
        # BCa can fail in degenerate cases — fall back to percentile
        result = scipy_bootstrap(
            (values,),
            statistic=lambda x, axis: np.apply_along_axis(metric_fn, axis, x),
            n_resamples=n_bootstrap,
            confidence_level=1.0 - alpha,
            method="percentile",
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_statistics.py::TestBCaBootstrap -v`
Expected: PASS

- [ ] **Step 6: Update paired_bootstrap_retention for BCa**

Replace `paired_bootstrap_retention` in `statistics.py`. **Critical**: use `paired=True` with
separate arrays, NOT column-stacking (column-stacking causes BCa to return NaN silently):

```python
def paired_bootstrap_retention(
    raw_scores: dict[str, float],
    comp_scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    """Bootstrap CI on the retention ratio: metric(compressed) / metric(raw).

    Uses paired resampling via scipy's paired=True — the same items are
    drawn for both raw and compressed in each bootstrap iteration, so
    correlated noise cancels. Uses BCa where possible (n >= 25).

    Args:
        raw_scores: {item_id: score} for the raw (uncompressed) system.
        comp_scores: {item_id: score} for the compressed system.
        metric_fn: Aggregation (default: np.mean).
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.
        alpha: Significance level.

    Returns:
        MetricResult where value is retention percentage (0-100+),
        ci_lower/ci_upper are the bootstrap BCa bounds.
    """
    from scipy.stats import bootstrap as scipy_bootstrap

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

    def retention_statistic(raw, comp, axis):
        raw_agg = np.apply_along_axis(metric_fn, axis, raw)
        comp_agg = np.apply_along_axis(metric_fn, axis, comp)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(raw_agg != 0, comp_agg / raw_agg * 100, 0.0)
        return ret

    use_bca = n >= 25
    method = "BCa" if use_bca else "percentile"

    def _run(m):
        return scipy_bootstrap(
            (raw_arr, comp_arr),
            statistic=retention_statistic,
            n_resamples=n_bootstrap,
            confidence_level=1.0 - alpha,
            method=m,
            paired=True,
            random_state=np.random.RandomState(seed),
        )

    try:
        result = _run(method)
        ci_lower = float(result.confidence_interval.low)
        ci_upper = float(result.confidence_interval.high)
        # Check for NaN (BCa can silently fail)
        if np.isnan(ci_lower) or np.isnan(ci_upper):
            raise ValueError("BCa returned NaN")
        ci_method = method.lower()
    except Exception:
        result = _run("percentile")
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
```

- [ ] **Step 7: Update paired_bootstrap_metric for BCa**

Replace `paired_bootstrap_metric` in `statistics.py`. Uses `paired=True` with separate
arrays to ensure BCa works correctly (same fix as Step 6):

```python
def paired_bootstrap_metric(
    raw_scores: dict[str, float],
    comp_scores: dict[str, float],
    metric_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[MetricResult, MetricResult]:
    """Bootstrap CIs on two systems evaluated on the same items (BCa).

    Computes independent BCa CIs for raw and compressed using the same
    set of items (aligned on common IDs). Each system gets its own
    bootstrap distribution. Use paired_bootstrap_retention for a CI on
    the ratio itself.

    Returns:
        (raw_metric, comp_metric) — both MetricResult with BCa CIs.
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    common = sorted(set(raw_scores.keys()) & set(comp_scores.keys()))
    if len(common) == 0:
        empty = MetricResult(value=0.0, ci_lower=0.0, ci_upper=0.0, n=0)
        return empty, empty

    raw_arr = np.array([raw_scores[k] for k in common], dtype=np.float64)
    comp_arr = np.array([comp_scores[k] for k in common], dtype=np.float64)
    n = len(common)

    raw_obs = float(metric_fn(raw_arr))
    comp_obs = float(metric_fn(comp_arr))

    use_bca = n >= 25
    method = "BCa" if use_bca else "percentile"

    def _bootstrap_one(arr, m):
        return scipy_bootstrap(
            (arr,),
            statistic=lambda x, axis: np.apply_along_axis(metric_fn, axis, x),
            n_resamples=n_bootstrap,
            confidence_level=1.0 - alpha,
            method=m,
            random_state=np.random.RandomState(seed),
        )

    try:
        raw_result = _bootstrap_one(raw_arr, method)
        comp_result = _bootstrap_one(comp_arr, method)
        raw_lo = float(raw_result.confidence_interval.low)
        raw_hi = float(raw_result.confidence_interval.high)
        comp_lo = float(comp_result.confidence_interval.low)
        comp_hi = float(comp_result.confidence_interval.high)
        if any(np.isnan(v) for v in [raw_lo, raw_hi, comp_lo, comp_hi]):
            raise ValueError("BCa returned NaN")
        ci_method = method.lower()
    except Exception:
        raw_result = _bootstrap_one(raw_arr, "percentile")
        comp_result = _bootstrap_one(comp_arr, "percentile")
        raw_lo = float(raw_result.confidence_interval.low)
        raw_hi = float(raw_result.confidence_interval.high)
        comp_lo = float(comp_result.confidence_interval.low)
        comp_hi = float(comp_result.confidence_interval.high)
        ci_method = "percentile"

    raw_mr = MetricResult(
        value=raw_obs, ci_lower=raw_lo, ci_upper=raw_hi,
        n=n, ci_method=ci_method,
    )
    comp_mr = MetricResult(
        value=comp_obs, ci_lower=comp_lo, ci_upper=comp_hi,
        n=n, ci_method=ci_method,
    )
    return raw_mr, comp_mr
```

- [ ] **Step 8: Fix existing tests for ci_method field**

Update `tests/test_benchmark_rules.py` to expect `ci_method`:

```python
class TestRule4StatisticalSignificance:
    """Rule 4: Every metric must have CI."""

    def test_metric_result_has_ci(self):
        r = MetricResult(value=0.95, ci_lower=0.93, ci_upper=0.97, n=1000)
        assert r.ci_lower < r.value < r.ci_upper
        assert r.ci_method == "percentile"  # default

    def test_metric_result_rejects_no_ci(self):
        with pytest.raises(ValueError):
            MetricResult(value=0.95, ci_lower=None, ci_upper=None, n=1000)

    def test_metric_result_bca(self):
        r = MetricResult(value=0.95, ci_lower=0.93, ci_upper=0.97, n=1000, ci_method="bca")
        assert r.ci_method == "bca"
```

- [ ] **Step 9: Run full test suite to verify nothing broke**

Run: `uv run pytest tests/test_benchmark_statistics.py tests/test_benchmark_rules.py -v`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add experiments/43_rigorous_benchmark/metrics/statistics.py \
       experiments/43_rigorous_benchmark/rules.py \
       tests/test_benchmark_statistics.py \
       tests/test_benchmark_rules.py
git commit -m "feat(exp43): upgrade bootstrap from percentile to BCa (DiCiccio & Efron 1996)

Switch all bootstrap CIs from percentile method (first-order accurate) to
BCa (second-order accurate, O(1/n) coverage error). Uses scipy.stats.bootstrap
internally. Falls back to percentile for n<25 where jackknife is unstable.

Adds ci_method field to MetricResult for transparency."
```

---

## Task 2: Averaged Multi-Seed Aggregation

**Files:**
- Modify: `experiments/43_rigorous_benchmark/metrics/statistics.py:172-201`
- Modify: `experiments/43_rigorous_benchmark/runners/per_residue.py:257-294`
- Test: `tests/test_benchmark_statistics.py`

### Rationale
Current approach: train 3 probes, pick median seed, report that seed's CI. This ignores probe training variability. Better: average predictions across seeds, then bootstrap once on the averaged per-protein scores. This reduces probe variance and captures both sources of uncertainty (Bouthillier et al. 2021).

- [ ] **Step 1: Write failing tests for averaged multi-seed**

In `tests/test_benchmark_statistics.py`, add:

```python
class TestAveragedMultiSeed:

    def test_averaged_seed_returns_metric_result(self):
        """Average per-item scores across seeds, then bootstrap."""
        seed_scores = [
            {f"p{i}": float(i * 0.1 + s * 0.01) for i in range(100)}
            for s in range(3)
        ]
        result = averaged_multi_seed(seed_scores, n_bootstrap=2000, seed=42)
        assert isinstance(result, MetricResult)
        assert result.seeds_mean is not None
        assert result.seeds_std is not None

    def test_averaged_reduces_variance(self):
        """Averaging across seeds should reduce noise vs single seed."""
        rng = np.random.RandomState(42)
        base = {f"p{i}": float(rng.rand()) for i in range(100)}
        # Each seed adds noise
        seed_scores = [
            {k: v + rng.randn() * 0.05 for k, v in base.items()}
            for _ in range(3)
        ]
        result = averaged_multi_seed(seed_scores, n_bootstrap=5000, seed=42)
        # Averaged should be close to base mean
        assert abs(result.value - np.mean(list(base.values()))) < 0.05

    def test_averaged_ci_is_bca(self):
        seed_scores = [
            {f"p{i}": float(i * 0.1) for i in range(100)}
            for _ in range(3)
        ]
        result = averaged_multi_seed(seed_scores, n_bootstrap=2000, seed=42)
        assert result.ci_method == "bca"

    def test_averaged_reports_per_seed_values(self):
        seed_scores = [
            {f"p{i}": 0.9 for i in range(100)},
            {f"p{i}": 0.95 for i in range(100)},
            {f"p{i}": 1.0 for i in range(100)},
        ]
        result = averaged_multi_seed(seed_scores, n_bootstrap=2000, seed=42)
        assert abs(result.seeds_mean - 0.95) < 0.01
        assert result.seeds_std > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_statistics.py::TestAveragedMultiSeed -v`
Expected: FAIL — `averaged_multi_seed` doesn't exist

- [ ] **Step 3: Implement averaged_multi_seed**

Add to `experiments/43_rigorous_benchmark/metrics/statistics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_statistics.py::TestAveragedMultiSeed -v`
Expected: PASS

- [ ] **Step 5: Update top-level imports in per_residue.py**

In `experiments/43_rigorous_benchmark/runners/per_residue.py`, update line 16:

```python
from metrics.statistics import bootstrap_ci, multi_seed_summary, averaged_multi_seed, cluster_bootstrap_ci
```

- [ ] **Step 6: Update SS3 runner to use averaged multi-seed**

In `experiments/43_rigorous_benchmark/runners/per_residue.py`, modify `run_ss3_benchmark` to collect per-protein scores from all seeds, then call `averaged_multi_seed`:

Replace lines 257-294 with:

```python
    seed_per_protein_scores = []  # list of {pid: q3} dicts
    seed_probe_results = []

    for seed in seeds:
        probe_result = train_classification_probe(
            X_train, y_train, X_test, y_test,
            C_grid=C_grid, cv_folds=cv_folds, seed=seed,
        )
        seed_probe_results.append(probe_result)

        fitted_probe = _get_fitted_logreg(
            X_train, y_train, probe_result["best_C"], seed
        )
        per_protein_scores = _per_protein_q3(
            embeddings, labels, test_ids, fitted_probe, max_len, SS3_MAP
        )
        seed_per_protein_scores.append(per_protein_scores)

    # Average across seeds, then bootstrap (Bouthillier et al. 2021)
    q3 = averaged_multi_seed(
        seed_per_protein_scores, n_bootstrap=n_bootstrap, seed=seeds[0]
    )

    # Select median-performing seed for per_class_acc and best_C reporting
    per_seed_means = [np.mean(list(s.values())) for s in seed_per_protein_scores]
    sorted_idx = np.argsort(per_seed_means)
    median_idx = int(sorted_idx[len(sorted_idx) // 2])
    median_probe = seed_probe_results[median_idx]

    return {
        "q3": q3,
        "per_class_acc": median_probe["per_class_acc"],
        "class_balance": class_balance,
        "best_C": median_probe["best_C"],
        "n_train_residues": int(len(y_train)),
        "n_test_residues": int(len(y_test)),
    }
```

- [ ] **Step 7: Apply same pattern to SS8 runner**

Mirror the SS3 changes in `run_ss8_benchmark` (lines 340-371): collect `seed_per_protein_scores`, call `averaged_multi_seed`, select median probe for reporting.

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest tests/test_benchmark_statistics.py tests/test_benchmark_rules.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add experiments/43_rigorous_benchmark/metrics/statistics.py \
       experiments/43_rigorous_benchmark/runners/per_residue.py \
       tests/test_benchmark_statistics.py
git commit -m "feat(exp43): average predictions across seeds before bootstrapping

Replace median-seed CI with averaged multi-seed approach: average
per-item predictions across 3 seeds, then compute one BCa bootstrap CI
on the averaged scores. Reports per-seed aggregate stats for transparency.

This captures both probe training variance (via averaging) and sampling
variance (via bootstrap), per Bouthillier et al. (2021, MLSys)."
```

---

## Task 3: Disorder Evaluation — Pooled Spearman ρ + AUC-ROC

**Files:**
- Modify: `experiments/43_rigorous_benchmark/runners/per_residue.py:378-462`
- Create: `tests/test_benchmark_disorder.py`
- Modify: `experiments/43_rigorous_benchmark/metrics/statistics.py` (add `cluster_bootstrap_ci`)

### Rationale
Every published disorder predictor (SETH, ODiNPred, ADOPT, UdonPred) reports **pooled residue-level Spearman ρ** — NOT per-protein averaged ρ. Our per-protein ρ=0.333 with CIs [76.6%, 120.0%] is both non-standard and uninformative due to extreme width. The fix: use pooled ρ as headline, with cluster bootstrap (resample proteins, recompute pooled ρ) for proper CIs that respect the hierarchical structure.

AUC-ROC on binary Z<8 threshold is the standard secondary metric (Nielsen & Mulder 2019, CAID).

- [ ] **Step 1: Write cluster_bootstrap_ci tests**

Create `tests/test_benchmark_disorder.py`:

```python
"""Tests for disorder evaluation methodology — Nature-level rigor.

Validates:
1. Cluster bootstrap for pooled metrics (resample proteins, compute pooled stat)
2. Pooled Spearman rho as headline (matching SETH/ODiNPred/ADOPT/UdonPred)
3. AUC-ROC on binary Z<8 threshold
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from rules import MetricResult
from metrics.statistics import cluster_bootstrap_ci


class TestClusterBootstrapCI:
    """Cluster bootstrap: resample clusters (proteins), compute pooled stat."""

    def test_returns_metric_result(self):
        rng = np.random.RandomState(42)
        # 50 proteins, each with 10-20 residues
        clusters = {
            f"p{i}": {
                "y_true": rng.rand(rng.randint(10, 21)),
                "y_pred": rng.rand(rng.randint(10, 21)),
            }
            for i in range(50)
        }
        # Make y_true and y_pred same length per protein
        for k, v in clusters.items():
            n = min(len(v["y_true"]), len(v["y_pred"]))
            clusters[k] = {"y_true": v["y_true"][:n], "y_pred": v["y_pred"][:n]}

        from scipy.stats import spearmanr
        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        result = cluster_bootstrap_ci(clusters, pooled_spearman,
                                       n_bootstrap=2000, seed=42)
        assert isinstance(result, MetricResult)
        assert result.ci_lower <= result.value <= result.ci_upper

    def test_ci_width_reasonable(self):
        """Cluster bootstrap CI should be narrower than per-protein CI."""
        rng = np.random.RandomState(42)
        # Correlated data: pred = true + noise
        clusters = {}
        for i in range(100):
            n = rng.randint(50, 200)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.5
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        from scipy.stats import spearmanr
        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho)

        result = cluster_bootstrap_ci(clusters, pooled_spearman,
                                       n_bootstrap=5000, seed=42)
        ci_width = result.ci_upper - result.ci_lower
        # For n=100 clusters with ~100 residues each, CI should be tight
        assert ci_width < 0.10  # Less than 0.1 wide

    def test_uses_bca_for_large_n(self):
        """Cluster bootstrap should use BCa for n >= 25 clusters, with pooled Spearman."""
        rng = np.random.RandomState(42)
        clusters = {}
        for i in range(50):
            n = rng.randint(20, 100)
            true = rng.randn(n)
            pred = true + rng.randn(n) * 0.3  # correlated predictions
            clusters[f"p{i}"] = {"y_true": true, "y_pred": pred}

        from scipy.stats import spearmanr
        def pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        result = cluster_bootstrap_ci(clusters, pooled_spearman,
                                       n_bootstrap=2000, seed=42)
        assert result.ci_method == "bca"
        # Pooled spearman should be positive for correlated data
        assert result.value > 0.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_benchmark_disorder.py -v`
Expected: FAIL — `cluster_bootstrap_ci` doesn't exist

- [ ] **Step 3: Implement cluster_bootstrap_ci**

Add to `experiments/43_rigorous_benchmark/metrics/statistics.py`:

```python
def cluster_bootstrap_ci(
    clusters: dict[str, dict],
    statistic_fn: Callable[[list[dict]], float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    """Cluster bootstrap CI for pooled metrics (e.g., pooled Spearman rho).

    Resamples at the cluster (protein) level, not the observation (residue)
    level, because residues within a protein are correlated. Each bootstrap
    iteration draws n clusters with replacement and computes the pooled
    statistic over all residues from the selected clusters.

    Reference: Davison & Hinkley (1997) Ch. 2.4; Field & Welsh (2007).

    Args:
        clusters: {cluster_id: dict} — each dict contains the data for one
            cluster (e.g., {"y_true": array, "y_pred": array}).
        statistic_fn: Takes list[dict] (selected cluster data), returns float.
            Example: pool all y_true/y_pred arrays, compute Spearman rho.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        alpha: Significance level.

    Returns:
        MetricResult with BCa CI (or percentile fallback for n < 25 clusters).
    """
    cluster_ids = sorted(clusters.keys())
    n = len(cluster_ids)
    cluster_data_list = [clusters[cid] for cid in cluster_ids]

    # Observed statistic
    observed = float(statistic_fn(cluster_data_list))

    # Manual bootstrap since scipy.stats.bootstrap doesn't handle dict clusters
    rng = np.random.RandomState(seed)
    boot_values = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        selected = [cluster_data_list[i] for i in idx]
        boot_values[b] = statistic_fn(selected)

    # BCa correction (manual implementation for cluster bootstrap)
    use_bca = n >= 25

    if use_bca:
        try:
            # Bias correction: z0
            z0 = _norm_ppf(np.mean(boot_values < observed))

            # Acceleration: jackknife
            jack_values = np.empty(n, dtype=np.float64)
            for i in range(n):
                jack_data = [cluster_data_list[j] for j in range(n) if j != i]
                jack_values[i] = statistic_fn(jack_data)

            jack_mean = np.mean(jack_values)
            num = np.sum((jack_mean - jack_values) ** 3)
            denom = 6.0 * (np.sum((jack_mean - jack_values) ** 2) ** 1.5)
            a_hat = num / denom if denom != 0 else 0.0

            # Adjusted percentiles
            z_alpha_lo = _norm_ppf(alpha / 2)
            z_alpha_hi = _norm_ppf(1 - alpha / 2)

            p_lo = _norm_cdf(z0 + (z0 + z_alpha_lo) / (1 - a_hat * (z0 + z_alpha_lo)))
            p_hi = _norm_cdf(z0 + (z0 + z_alpha_hi) / (1 - a_hat * (z0 + z_alpha_hi)))

            ci_lower = float(np.percentile(boot_values, 100 * p_lo))
            ci_upper = float(np.percentile(boot_values, 100 * p_hi))
            ci_method = "bca"
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # Fallback to percentile
            ci_lower = float(np.percentile(boot_values, 100 * alpha / 2))
            ci_upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
            ci_method = "percentile"
    else:
        ci_lower = float(np.percentile(boot_values, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
        ci_method = "percentile"

    return MetricResult(
        value=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=n,
        ci_method=ci_method,
    )


def _norm_ppf(p: float) -> float:
    """Normal distribution percent point function (inverse CDF)."""
    from scipy.stats import norm
    # Clamp to avoid inf
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(norm.ppf(p))


def _norm_cdf(z: float) -> float:
    """Normal distribution CDF."""
    from scipy.stats import norm
    return float(norm.cdf(z))
```

- [ ] **Step 4: Run cluster bootstrap tests**

Run: `uv run pytest tests/test_benchmark_disorder.py -v`
Expected: PASS

- [ ] **Step 5: Write tests for the updated disorder runner**

Add to `tests/test_benchmark_disorder.py`:

```python
class TestDisorderBenchmarkRunner:
    """Test the updated disorder runner with pooled ρ + AUC-ROC."""

    def _make_disorder_data(self, n_proteins=30, seq_len=50, seed=42):
        """Create synthetic disorder data for testing."""
        rng = np.random.RandomState(seed)
        embeddings = {}
        scores = {}
        for i in range(n_proteins):
            L = rng.randint(seq_len // 2, seq_len * 2)
            embeddings[f"p{i}"] = rng.randn(L, 64).astype(np.float32)
            # Z-scores: mix of ordered (>8) and disordered (<8)
            z = rng.randn(L) * 5 + 10  # mean ~10, some < 8
            scores[f"p{i}"] = z

        train_ids = [f"p{i}" for i in range(0, n_proteins, 2)]
        test_ids = [f"p{i}" for i in range(1, n_proteins, 2)]
        return embeddings, scores, train_ids, test_ids

    def test_returns_pooled_rho_as_headline(self):
        from runners.per_residue import run_disorder_benchmark
        emb, sc, train, test = self._make_disorder_data()
        result = run_disorder_benchmark(
            emb, sc, train, test,
            seeds=[42], n_bootstrap=500, max_len=100,
        )
        assert "pooled_spearman_rho" in result
        assert isinstance(result["pooled_spearman_rho"], MetricResult)
        assert result["pooled_spearman_rho"].ci_lower is not None

    def test_returns_auc_roc(self):
        from runners.per_residue import run_disorder_benchmark
        emb, sc, train, test = self._make_disorder_data()
        result = run_disorder_benchmark(
            emb, sc, train, test,
            seeds=[42], n_bootstrap=500, max_len=100,
        )
        assert "auc_roc" in result
        assert isinstance(result["auc_roc"], MetricResult)
        assert 0.0 <= result["auc_roc"].value <= 1.0

    def test_per_protein_rho_is_supplementary(self):
        from runners.per_residue import run_disorder_benchmark
        emb, sc, train, test = self._make_disorder_data()
        result = run_disorder_benchmark(
            emb, sc, train, test,
            seeds=[42], n_bootstrap=500, max_len=100,
        )
        # Per-protein should still exist as supplementary
        assert "per_protein_spearman_rho" in result
        assert isinstance(result["per_protein_spearman_rho"], MetricResult)
```

- [ ] **Step 6: Run to verify they fail**

Run: `uv run pytest tests/test_benchmark_disorder.py::TestDisorderBenchmarkRunner -v`
Expected: FAIL — disorder runner doesn't return `pooled_spearman_rho` as MetricResult or `auc_roc`

- [ ] **Step 7: Rewrite run_disorder_benchmark**

Replace `run_disorder_benchmark` in `runners/per_residue.py`:

```python
def run_disorder_benchmark(
    embeddings: dict,
    scores: dict,
    train_ids: list,
    test_ids: list,
    alpha_grid: list = None,
    cv_folds: int = 3,
    seeds: list = None,
    n_bootstrap: int = 10_000,
    max_len: int = 512,
    disorder_threshold: float = 8.0,
) -> dict:
    """Run disorder benchmark with pooled Spearman rho + AUC-ROC (literature standard).

    Headline metric: pooled residue-level Spearman rho with cluster bootstrap CI
    (resample proteins, recompute pooled rho). This matches the evaluation
    methodology of SETH, ODiNPred, ADOPT, UdonPred, and the CheZOD benchmark
    (Nielsen & Mulder 2019).

    Secondary metric: AUC-ROC on binary Z < threshold (standard in CAID).

    Supplementary: per-protein Spearman rho with BCa CI (conservative, for
    transparency only).

    Args:
        embeddings: {protein_id: (L, D) float32 array}
        scores: {protein_id: (L,) float array with possible NaN — Z-scores}
        train_ids: Protein IDs for training.
        test_ids: Protein IDs for testing.
        alpha_grid: Ridge regularization grid.
        cv_folds: Number of CV folds.
        seeds: List of random seeds.
        n_bootstrap: Bootstrap iterations for CI.
        max_len: Maximum residues per protein.
        disorder_threshold: Z-score threshold for binary classification (default 8.0,
            per Nielsen & Mulder 2019).

    Returns:
        dict with:
            "pooled_spearman_rho": MetricResult — headline, cluster bootstrap CI
            "auc_roc": MetricResult — binary disorder AUC, cluster bootstrap CI
            "per_protein_spearman_rho": MetricResult — supplementary, BCa CI
            "best_alpha": float from median seed
            "n_train_residues": int
            "n_test_residues": int
    """
    from scipy.stats import spearmanr as _spearmanr
    from sklearn.metrics import roc_auc_score

    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    if seeds is None:
        seeds = [42, 43, 44]

    check_no_leakage(train_ids, test_ids)

    X_train, y_train = _stack_residues(embeddings, scores, train_ids, max_len, label_map=None)
    X_test, y_test = _stack_residues(embeddings, scores, test_ids, max_len, label_map=None)

    # ---- Train probes across seeds, collect per-protein predictions ----
    seed_cluster_data = []  # list of {pid: {"y_true": arr, "y_pred": arr}}
    seed_per_protein_rhos = []  # for supplementary per-protein metric
    seed_probe_results = []

    for seed in seeds:
        probe_result = train_regression_probe(
            X_train, y_train, X_test, y_test,
            alpha_grid=alpha_grid, cv_folds=cv_folds, seed=seed,
        )
        seed_probe_results.append(probe_result)

        # Collect per-protein predictions for cluster bootstrap
        fitted_probe = _get_fitted_ridge(X_train, y_train, probe_result["best_alpha"])
        cluster_data = {}
        per_protein_rhos = {}

        for pid in test_ids:
            if pid not in embeddings or pid not in scores:
                continue
            emb = embeddings[pid][:max_len]
            lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
            n = min(len(emb), len(lab))

            X_p, y_p = [], []
            for i in range(n):
                if not np.isnan(lab[i]):
                    X_p.append(emb[i])
                    y_p.append(lab[i])

            if len(X_p) < 5:
                continue

            X_p = np.stack(X_p).astype(np.float32)
            y_p = np.array(y_p, dtype=np.float64)
            preds = fitted_probe.predict(X_p)

            cluster_data[pid] = {"y_true": y_p, "y_pred": preds}
            rho, _ = _spearmanr(y_p, preds)
            if not np.isnan(rho):
                per_protein_rhos[pid] = float(rho)

        seed_cluster_data.append(cluster_data)
        seed_per_protein_rhos.append(per_protein_rhos)

    # ---- Average predictions across seeds (per-protein, per-residue) ----
    all_pids = sorted(set.intersection(*[set(cd.keys()) for cd in seed_cluster_data]))

    averaged_clusters = {}
    for pid in all_pids:
        y_true = seed_cluster_data[0][pid]["y_true"]  # same across seeds
        y_preds = [seed_cluster_data[s][pid]["y_pred"] for s in range(len(seeds))]
        y_pred_avg = np.mean(y_preds, axis=0)
        averaged_clusters[pid] = {"y_true": y_true, "y_pred": y_pred_avg}

    # ---- Headline: Pooled Spearman rho with cluster bootstrap ----
    def pooled_spearman(cluster_data_list):
        all_true = np.concatenate([d["y_true"] for d in cluster_data_list])
        all_pred = np.concatenate([d["y_pred"] for d in cluster_data_list])
        rho, _ = _spearmanr(all_true, all_pred)
        return float(rho) if not np.isnan(rho) else 0.0

    pooled_rho_result = cluster_bootstrap_ci(
        averaged_clusters, pooled_spearman,
        n_bootstrap=n_bootstrap, seed=seeds[0],
    )

    # ---- Secondary: AUC-ROC on binary Z < threshold ----
    def pooled_auc(cluster_data_list):
        all_true = np.concatenate([d["y_true"] for d in cluster_data_list])
        all_pred = np.concatenate([d["y_pred"] for d in cluster_data_list])
        # Binary labels: Z < threshold = disordered (positive class)
        y_binary = (all_true < disorder_threshold).astype(int)
        # Lower predicted Z-score = more disordered = higher "disorder score"
        # Negate predictions so higher = more disordered
        disorder_score = -all_pred
        if len(np.unique(y_binary)) < 2:
            return 0.5  # No positive or negative examples
        return float(roc_auc_score(y_binary, disorder_score))

    auc_result = cluster_bootstrap_ci(
        averaged_clusters, pooled_auc,
        n_bootstrap=n_bootstrap, seed=seeds[0],
    )

    # ---- Supplementary: per-protein rho with averaged seeds ----
    per_protein_rho_result = averaged_multi_seed(
        seed_per_protein_rhos, n_bootstrap=n_bootstrap, seed=seeds[0],
    )

    # Median seed for reporting best_alpha
    per_seed_pooled = []
    for cd in seed_cluster_data:
        data_list = [cd[pid] for pid in all_pids if pid in cd]
        per_seed_pooled.append(pooled_spearman(data_list))
    sorted_idx = np.argsort(per_seed_pooled)
    median_idx = int(sorted_idx[len(sorted_idx) // 2])
    median_probe = seed_probe_results[median_idx]

    return {
        "pooled_spearman_rho": pooled_rho_result,
        "auc_roc": auc_result,
        "per_protein_spearman_rho": per_protein_rho_result,
        # Backward compat: "spearman_rho" aliases "per_protein_spearman_rho"
        # so run_phase_a1.py and run_phase_b.py don't break
        "spearman_rho": per_protein_rho_result,
        "best_alpha": median_probe["best_alpha"],
        "n_train_residues": int(len(y_train)),
        "n_test_residues": int(len(y_test)),
        "n_test_proteins": len(all_pids),
        "disorder_threshold": disorder_threshold,
    }
```

- [ ] **Step 8: Update phase scripts for pooled_spearman_rho type change**

The return dict now has `pooled_spearman_rho` as MetricResult (was bare float).
Phase scripts use `:.4f` formatting which crashes on MetricResult. Fix all 7 call sites:

In `experiments/43_rigorous_benchmark/run_phase_a1.py`, change:
- Line 388: `{dis_raw['pooled_spearman_rho']:.4f}` → `{dis_raw['pooled_spearman_rho'].value:.4f}`
- Line 418: `{dis_comp['pooled_spearman_rho']:.4f}` → `{dis_comp['pooled_spearman_rho'].value:.4f}`
- Line 420: `fmt_retention("...", dis_comp["pooled_spearman_rho"], dis_raw["pooled_spearman_rho"])` → `fmt_retention("...", dis_comp["pooled_spearman_rho"].value, dis_raw["pooled_spearman_rho"].value)`

In `experiments/43_rigorous_benchmark/run_phase_b.py`, change:
- Line 707: `{dis_raw_chezod['pooled_spearman_rho']:.4f}` → `{dis_raw_chezod['pooled_spearman_rho'].value:.4f}`
- Line 720: `{dis_comp_chezod['pooled_spearman_rho']:.4f}` → `{dis_comp_chezod['pooled_spearman_rho'].value:.4f}`
- Line 799: `{dis_raw_trizod['pooled_spearman_rho']:.4f}` → `{dis_raw_trizod['pooled_spearman_rho'].value:.4f}`
- Line 812: `{dis_comp_trizod['pooled_spearman_rho']:.4f}` → `{dis_comp_trizod['pooled_spearman_rho'].value:.4f}`

- [ ] **Step 9: Run disorder tests**

Run: `uv run pytest tests/test_benchmark_disorder.py -v`
Expected: PASS

- [ ] **Step 10: Run full test suite**

Run: `uv run pytest tests/test_benchmark_statistics.py tests/test_benchmark_rules.py tests/test_benchmark_disorder.py -v`
Expected: ALL PASS

- [ ] **Step 11: Commit**

```bash
git add experiments/43_rigorous_benchmark/metrics/statistics.py \
       experiments/43_rigorous_benchmark/runners/per_residue.py \
       experiments/43_rigorous_benchmark/run_phase_a1.py \
       experiments/43_rigorous_benchmark/run_phase_b.py \
       tests/test_benchmark_disorder.py
git commit -m "feat(exp43): disorder evaluation to pooled Spearman rho + AUC-ROC

Switch disorder headline metric from per-protein averaged rho (non-standard,
wide CIs) to pooled residue-level Spearman rho (matching SETH, ODiNPred,
ADOPT, UdonPred literature standard). Add AUC-ROC on binary Z<8 threshold
(CAID standard).

Cluster bootstrap (resample proteins, compute pooled stat) respects the
hierarchical data structure (Davison & Hinkley 1997, Ch. 2.4).

Per-protein rho retained as supplementary metric for transparency."
```

---

## Task 4: ABTT Cross-Corpus Stability Test

**Files:**
- Create: `experiments/43_rigorous_benchmark/metrics/abtt_stability.py`
- Create: `tests/test_abtt_stability.py`

### Rationale
ABTT is fitted on SCOPe 5K, but SCOPe 5K is also used for retrieval evaluation (confirmed overlap in Phases A1, B-ESM2, D). Rather than breaking the overlap (which would reduce the retrieval test set), we prove it doesn't matter: the top-3 PCs are properties of the PLM architecture, not the data. We fit ABTT on 3+ corpora and show: (a) PCs are nearly identical via principal angles, (b) downstream performance is stable within noise. No protein embedding paper has done this — novel methodological contribution.

References: Bjorck & Golub (1973) for principal angles; Mu & Viswanath (2018) for ABTT.

- [ ] **Step 1: Write tests for principal angle computation**

Create `tests/test_abtt_stability.py`:

```python
"""Tests for ABTT cross-corpus stability analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "43_rigorous_benchmark"))

import numpy as np
import pytest
from metrics.abtt_stability import principal_angles, subspace_similarity


class TestPrincipalAngles:
    """Test Bjorck & Golub (1973) principal angle computation."""

    def test_identical_subspaces_zero_angles(self):
        """Same subspace should have zero principal angles."""
        rng = np.random.RandomState(42)
        A = rng.randn(3, 100).astype(np.float32)
        angles = principal_angles(A, A)
        np.testing.assert_allclose(angles, 0.0, atol=1e-5)

    def test_orthogonal_subspaces_90_degrees(self):
        """Orthogonal subspaces should have 90-degree angles."""
        # First 3 dims vs next 3 dims in 100d space
        A = np.eye(100, dtype=np.float32)[:3]  # rows 0,1,2
        B = np.eye(100, dtype=np.float32)[3:6]  # rows 3,4,5
        angles = principal_angles(A, B)
        np.testing.assert_allclose(angles, np.pi / 2, atol=1e-5)

    def test_returns_sorted_angles(self):
        rng = np.random.RandomState(42)
        A = rng.randn(3, 100).astype(np.float32)
        B = rng.randn(3, 100).astype(np.float32)
        angles = principal_angles(A, B)
        assert len(angles) == 3
        assert all(angles[i] <= angles[i + 1] for i in range(len(angles) - 1))

    def test_angles_between_0_and_pi_half(self):
        rng = np.random.RandomState(42)
        A = rng.randn(3, 100).astype(np.float32)
        B = rng.randn(3, 100).astype(np.float32)
        angles = principal_angles(A, B)
        assert all(0 <= a <= np.pi / 2 + 1e-6 for a in angles)


class TestSubspaceSimilarity:

    def test_identical_is_one(self):
        rng = np.random.RandomState(42)
        A = rng.randn(3, 100).astype(np.float32)
        sim = subspace_similarity(A, A)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_is_zero(self):
        A = np.eye(100, dtype=np.float32)[:3]
        B = np.eye(100, dtype=np.float32)[3:6]
        sim = subspace_similarity(A, B)
        assert abs(sim) < 1e-5

    def test_between_zero_and_one(self):
        rng = np.random.RandomState(42)
        A = rng.randn(3, 100).astype(np.float32)
        B = rng.randn(3, 100).astype(np.float32)
        sim = subspace_similarity(A, B)
        assert 0 <= sim <= 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_abtt_stability.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement ABTT stability functions**

Create `experiments/43_rigorous_benchmark/metrics/abtt_stability.py`:

```python
"""ABTT cross-corpus stability analysis.

Tests whether the top-k principal components removed by ABTT are properties
of the PLM architecture (stable across corpora) or data-dependent (potential
information leakage).

Novel contribution: no published protein embedding paper has tested this.

References:
    Bjorck & Golub (1973) "Numerical Methods for Computing Angles Between
        Linear Subspaces of R^n" — principal angles.
    Mu & Viswanath (2018) "All-but-the-Top" — ABTT.
"""

import numpy as np


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces (Bjorck & Golub 1973).

    Principal angles are the canonical angles between the column spaces of
    A.T and B.T, computed via the SVD of Q_A.T @ Q_B where Q_A and Q_B are
    orthonormal bases for the two subspaces.

    Args:
        A: (k, D) — rows are basis vectors for subspace A.
        B: (k, D) — rows are basis vectors for subspace B.

    Returns:
        (k,) array of principal angles in radians, sorted ascending [0, pi/2].
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # Orthonormalize rows via QR on the transpose
    Q_A, _ = np.linalg.qr(A.T, mode="reduced")  # (D, k)
    Q_B, _ = np.linalg.qr(B.T, mode="reduced")  # (D, k)

    # SVD of the cross-product matrix
    M = Q_A.T @ Q_B  # (k, k)
    _, sigmas, _ = np.linalg.svd(M)

    # Clamp to [0, 1] for numerical stability
    sigmas = np.clip(sigmas, 0.0, 1.0)

    # Principal angles = arccos(singular values), sorted ascending
    angles = np.sort(np.arccos(sigmas))

    return angles.astype(np.float64)


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Grassmann distance-based similarity between two subspaces.

    Similarity = mean(cos^2(principal_angles)). Returns 1.0 for identical
    subspaces, 0.0 for orthogonal subspaces.

    Args:
        A: (k, D) — rows are basis vectors for subspace A.
        B: (k, D) — rows are basis vectors for subspace B.

    Returns:
        Similarity score in [0, 1].
    """
    angles = principal_angles(A, B)
    return float(np.mean(np.cos(angles) ** 2))


def cross_corpus_stability_report(
    corpora: dict[str, np.ndarray],
    k: int = 3,
    seed: int = 42,
) -> dict:
    """Fit ABTT on multiple corpora and compare top-k PCs.

    For each corpus, fits ABTT (mean + top-k PCs), then computes pairwise
    subspace similarity between all pairs of corpora. A similarity close
    to 1.0 proves the PCs are PLM-architectural, not data-dependent.

    Args:
        corpora: {corpus_name: (N, D) stacked residue embeddings}
        k: Number of PCs to compare (default 3, matching ABTT-3).
        seed: Random seed for subsampling in fit_abtt.

    Returns:
        dict with:
            "params": {corpus_name: {"mean": (D,), "top_pcs": (k, D)}}
            "pairwise_similarity": {(name_a, name_b): float}
            "pairwise_angles_deg": {(name_a, name_b): list[float]}
            "min_similarity": float
            "mean_similarity": float
            "conclusion": str — "stable" if min_sim > 0.95, else "unstable"
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.one_embedding.core.preprocessing import fit_abtt

    # Fit ABTT on each corpus
    params = {}
    for name, residues in corpora.items():
        params[name] = fit_abtt(residues, k=k, seed=seed)

    # Pairwise comparison
    names = sorted(params.keys())
    pairwise_sim = {}
    pairwise_angles = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            angles = principal_angles(params[a]["top_pcs"], params[b]["top_pcs"])
            sim = subspace_similarity(params[a]["top_pcs"], params[b]["top_pcs"])
            pairwise_sim[(a, b)] = sim
            pairwise_angles[(a, b)] = [float(np.degrees(ang)) for ang in angles]

    sims = list(pairwise_sim.values())
    min_sim = min(sims) if sims else 1.0
    mean_sim = float(np.mean(sims)) if sims else 1.0

    # Stability threshold: 0.95 = principal angles all < ~13 degrees
    conclusion = "stable" if min_sim > 0.95 else "unstable"

    return {
        "params": params,
        "pairwise_similarity": pairwise_sim,
        "pairwise_angles_deg": pairwise_angles,
        "min_similarity": min_sim,
        "mean_similarity": mean_sim,
        "conclusion": conclusion,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_abtt_stability.py -v`
Expected: PASS

- [ ] **Step 5: Add cross-corpus stability test with synthetic data**

Add to `tests/test_abtt_stability.py`:

```python
class TestCrossCorpusStability:

    def test_same_distribution_high_similarity(self):
        """Corpora from the same distribution should yield similar PCs."""
        from metrics.abtt_stability import cross_corpus_stability_report
        rng = np.random.RandomState(42)
        # Generate from same distribution with VERY dominant top PCs
        # (mimics PLM behavior: top eigenvalues 100x larger than rest)
        eigenvalues = np.ones(64, dtype=np.float32)
        eigenvalues[:3] = 100.0  # Top 3 PCs are 100x dominant
        base_cov = np.diag(eigenvalues)

        corpora = {}
        for name in ["corpus_a", "corpus_b", "corpus_c"]:
            corpora[name] = (rng.randn(10000, 64) @ np.linalg.cholesky(base_cov).T
                             ).astype(np.float32)

        report = cross_corpus_stability_report(corpora, k=3, seed=42)
        # With 100x dominant PCs and 10K samples, subspaces should be very stable
        assert report["min_similarity"] > 0.95
        assert report["conclusion"] == "stable"

    def test_different_distributions_low_similarity(self):
        """Corpora from very different distributions should yield different PCs."""
        from metrics.abtt_stability import cross_corpus_stability_report
        rng = np.random.RandomState(42)

        corpora = {
            "uniform": rng.rand(5000, 64).astype(np.float32),
            "structured": (rng.randn(5000, 1) * np.arange(64)).astype(np.float32),
        }

        report = cross_corpus_stability_report(corpora, k=3, seed=42)
        # Very different distributions should give low similarity
        assert report["min_similarity"] < 0.90
```

- [ ] **Step 6: Run all stability tests**

Run: `uv run pytest tests/test_abtt_stability.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add experiments/43_rigorous_benchmark/metrics/abtt_stability.py \
       tests/test_abtt_stability.py
git commit -m "feat(exp43): ABTT cross-corpus stability test (Bjorck & Golub 1973)

Novel methodological contribution: prove that ABTT top-k PCs are properties
of the PLM architecture, not data-dependent. Uses principal angles between
subspaces fitted on different corpora. If PCs are stable (similarity > 0.95),
the choice of fitting corpus is irrelevant — making the SCOPe-on-SCOPe
overlap a non-issue.

No published protein embedding paper has tested this."
```

---

## Task 5: Update Retrieval Runner for BCa

**Files:**
- Modify: `experiments/43_rigorous_benchmark/runners/protein_level.py:85-107`
- Test: existing tests should pass (bootstrap_ci already upgraded)

The retrieval runner calls `bootstrap_ci` from statistics.py, which is now BCa. No code change needed in the runner itself — the upgrade propagates automatically. But verify.

- [ ] **Step 1: Verify retrieval runner uses upgraded bootstrap**

Run: `uv run pytest tests/test_benchmark_statistics.py -v`
Expected: ALL PASS (retrieval runner inherits BCa from bootstrap_ci)

- [ ] **Step 2: Add a test that retrieval bootstrap uses BCa**

Add to `tests/test_benchmark_statistics.py`:

```python
class TestRetrievalBCa:

    def test_retrieval_scores_get_bca(self):
        """Retrieval Ret@1 scores (binary 0/1) should get BCa CI."""
        rng = np.random.RandomState(42)
        scores = {f"q{i}": float(rng.choice([0.0, 1.0])) for i in range(200)}
        result = bootstrap_ci(scores, metric_fn=np.mean, n_bootstrap=2000, seed=42)
        assert result.ci_method == "bca"
        assert result.ci_lower <= result.value <= result.ci_upper
```

- [ ] **Step 3: Run test**

Run: `uv run pytest tests/test_benchmark_statistics.py::TestRetrievalBCa -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_benchmark_statistics.py
git commit -m "test(exp43): verify retrieval bootstrap uses BCa"
```

---

## Task 6: Remove Legacy multi_seed_summary

**Files:**
- Modify: `experiments/43_rigorous_benchmark/metrics/statistics.py`
- Modify: `experiments/43_rigorous_benchmark/runners/per_residue.py` (remove old imports)
- Modify: `tests/test_benchmark_statistics.py` (update tests)

### Rationale
`multi_seed_summary` (median seed CI) is now superseded by `averaged_multi_seed`. Keep `multi_seed_summary` but add a deprecation docstring — it may still be used by external code or old phase scripts.

- [ ] **Step 1: Add deprecation notice to multi_seed_summary**

```python
def multi_seed_summary(seed_results: list[MetricResult]) -> MetricResult:
    """Aggregate results across multiple seeds (Rule 5) — DEPRECATED.

    Prefer averaged_multi_seed() which averages per-item predictions
    across seeds before bootstrapping, properly capturing both probe
    training variance and sampling variance.

    This function selects the median-performing seed's CI, which ignores
    probe training variability (Bouthillier et al. 2021).
    """
    # ... (keep existing implementation)
```

- [ ] **Step 2: Ensure runners import averaged_multi_seed**

Verify the imports at the top of `per_residue.py` include `averaged_multi_seed`.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add experiments/43_rigorous_benchmark/metrics/statistics.py \
       experiments/43_rigorous_benchmark/runners/per_residue.py
git commit -m "refactor(exp43): deprecate multi_seed_summary in favor of averaged_multi_seed"
```

---

## Task 7: Full Test Suite Verification

**Files:** All test files

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest tests/ -v --timeout=120 2>&1 | tail -30`
Expected: ALL PASS (726+ tests)

- [ ] **Step 2: Verify new test count**

Run: `uv run pytest tests/ --co -q | tail -5`
Expected: ~750+ tests (726 original + ~24 new)

- [ ] **Step 3: Check for any import issues in benchmark framework**

Run: `uv run python -c "
import sys; sys.path.insert(0, 'experiments/43_rigorous_benchmark')
from metrics.statistics import bootstrap_ci, paired_bootstrap_retention, paired_bootstrap_metric, averaged_multi_seed, cluster_bootstrap_ci
from metrics.abtt_stability import principal_angles, subspace_similarity, cross_corpus_stability_report
from runners.per_residue import run_ss3_benchmark, run_ss8_benchmark, run_disorder_benchmark
from runners.protein_level import run_retrieval_benchmark
print('All imports OK')
"`
Expected: "All imports OK"

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test(exp43): verify full test suite passes with methodology hardening"
```

---

## Post-Implementation Notes

### What This Plan Does NOT Cover (deferred to separate plans)
1. **Re-running Phases A-D** with the new methodology — requires compute time, separate plan
2. **V2 extreme compression validation** — needs its own benchmark pass
3. **PQ at 768d** — new experiment, separate plan
4. **Updating CLAUDE.md numbers** — after benchmarks re-run
5. **CI/CD pipeline** — GitHub Actions, separate concern

### Verification Checklist for Re-run (Phase 4)
When re-running benchmarks after these code changes, verify:
- [ ] All CIs report `ci_method: "bca"` (except CASP12 n=20 which may fall back)
- [ ] Disorder headline is pooled ρ (comparable to SETH 0.72, ODiNPred 0.649)
- [ ] Disorder AUC-ROC is reported (comparable to SETH 0.91, ODiNPred 0.914)
- [ ] SS3/SS8 use averaged multi-seed (seeds_std should be < 0.005)
- [ ] ABTT stability report shows similarity > 0.95 across corpora
- [ ] All cross-dataset consistency checks pass (< 3pp divergence)

### Key Literature for Methods Section
- Bootstrap: Efron & Tibshirani (1993), DiCiccio & Efron (1996)
- Cluster bootstrap: Davison & Hinkley (1997), Field & Welsh (2007)
- Multi-seed: Bouthillier et al. (2021)
- Disorder evaluation: Nielsen & Mulder (2019), SETH (Ilzhoefer 2022)
- Principal angles: Bjorck & Golub (1973)
- ABTT: Mu & Viswanath (2018)
