"""Multi-seed, cluster-paired bootstrap for retention = metric_OE / metric_raw.

Design (fan-reviews #2/#3):
  * Paired: both arms must share identical test items per (task, seed); `item_ids_raw` must
    equal `item_ids_oe` (same ids, same order). The caller aligns by id first.
  * Cluster bootstrap: for per-residue tasks, residues within a protein are correlated, so
    resampling residues i.i.d. understates variance. Pass `group_ids` (e.g. protein id per
    residue) and the bootstrap resamples whole GROUPS with replacement. For per-sequence tasks
    leave `group_ids=None` (each item is its own group).
  * Multi-seed coherence: the bootstrapped statistic AVERAGES over all S probe seeds on the
    same resample (so it matches the point estimate = mean-over-seeds full-sample ratio). The
    probe-seed noise floor is reported separately as `between_seed_sd`; the decision rule
    (in 05_analyze) combines the item-variance CI with the seed standard error.
  * BCa (bias-correction + leave-one-GROUP-out jackknife acceleration) for both ratio and Δ.
"""
import numpy as np
from scipy.stats import spearmanr, norm


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.0
    rho = spearmanr(y_true, y_pred).correlation
    return float(rho) if rho == rho else 0.0  # NaN -> 0


_METRICS = {"accuracy": accuracy, "spearman": spearman}


def _grouped_indices(group_ids):
    """Map group id -> int array of member item indices (insertion order preserved)."""
    groups = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    keys = list(groups)
    return keys, {k: np.asarray(v, dtype=int) for k, v in groups.items()}


def _mean_seed_ratio(m, y_true, raw_preds, oe_preds, idx):
    S = raw_preds.shape[0]
    vals = np.empty(S)
    for s in range(S):
        mr = m(y_true[idx], raw_preds[s][idx])
        mo = m(y_true[idx], oe_preds[s][idx])
        vals[s] = mo / mr if mr != 0 else np.nan
    return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan


def _mean_seed_delta(m, y_true, raw_preds, oe_preds, idx):
    S = raw_preds.shape[0]
    vals = np.array([m(y_true[idx], oe_preds[s][idx]) - m(y_true[idx], raw_preds[s][idx]) for s in range(S)])
    return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan


def _bca_interval(boot, theta_hat, jack_values, alpha=0.05):
    """BCa from a bootstrap distribution + leave-one-out jackknife statistics."""
    boot = np.asarray(boot)
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return (float("nan"), float("nan"))
    prop = float(np.mean(boot < theta_hat))
    z0 = norm.ppf(prop) if 0.0 < prop < 1.0 else 0.0
    a = 0.0
    if jack_values is not None:
        jv = np.asarray(jack_values)
        jv = jv[~np.isnan(jv)]
        if jv.size > 1:
            jbar = jv.mean()
            num = np.sum((jbar - jv) ** 3)
            den = 6.0 * (np.sum((jbar - jv) ** 2) ** 1.5)
            a = num / den if den != 0 else 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def adj(z):
        denom = 1 - a * (z0 + z)
        return float(norm.cdf(z0 + (z0 + z) / denom)) if denom != 0 else float(norm.cdf(z0 + z))

    a1 = float(np.clip(adj(zl) * 100.0, 0.0, 100.0))
    a2 = float(np.clip(adj(zu) * 100.0, 0.0, 100.0))
    return float(np.percentile(boot, a1)), float(np.percentile(boot, a2))


def paired_retention(item_ids_raw, item_ids_oe, y_true, raw_preds, oe_preds,
                     group_ids=None, metric="accuracy", n_boot=2000, seed=42, acceleration=True):
    """Multi-seed cluster-paired bootstrap retention with BCa CI.

    Args:
        item_ids_raw, item_ids_oe: length-N id sequences; MUST be identical (same order).
        y_true: (N,) ground truth aligned to item order.
        raw_preds, oe_preds: (S, N) per-seed per-item predictions.
        group_ids: optional (N,) cluster ids (e.g. protein id for per-residue tasks). None ->
            each item is its own group (i.i.d. resample; correct for per-sequence tasks).
        acceleration: if True, BCa via leave-one-group-out jackknife; else BC (a=0).

    Returns dict: ratio, ci_low/ci_high (BCa, item variance), between_seed_sd (probe-noise
    floor), seed_ratios, delta, delta_ci (BCa), n_items/n_seeds/n_groups/n_boot.
    """
    if list(item_ids_raw) != list(item_ids_oe):
        raise ValueError("arms must share identical test items in the same order")
    m = _METRICS[metric]
    y_true = np.asarray(y_true)
    raw_preds = np.asarray(raw_preds)
    oe_preds = np.asarray(oe_preds)
    if raw_preds.ndim != 2 or oe_preds.shape != raw_preds.shape:
        raise ValueError(f"raw_preds/oe_preds must both be (S, N); got {raw_preds.shape}, {oe_preds.shape}")
    S, N = raw_preds.shape
    if len(y_true) != N or len(item_ids_raw) != N:
        raise ValueError("y_true/item_ids length must match N (=raw_preds.shape[1])")
    if group_ids is None:
        group_ids = np.arange(N)
    elif len(group_ids) != N:
        raise ValueError("group_ids length must match N")
    gkeys, gmap = _grouped_indices(group_ids)
    n_groups = len(gkeys)

    # Probe-noise floor: SD of full-sample per-seed ratios (guard on non-NaN count).
    seed_ratios = np.array([
        (lambda mr, mo: mo / mr if mr != 0 else np.nan)(m(y_true, raw_preds[s]), m(y_true, oe_preds[s]))
        for s in range(S)
    ])
    valid_sr = seed_ratios[~np.isnan(seed_ratios)]
    theta_hat = float(np.nanmean(seed_ratios)) if valid_sr.size else float("nan")
    between_seed_sd = float(np.std(valid_sr, ddof=1)) if valid_sr.size > 1 else 0.0
    delta_hat = _mean_seed_delta(m, y_true, raw_preds, oe_preds, np.arange(N))
    # Mean-over-seeds full-sample metrics (for the small-denominator check in 05_analyze).
    metric_raw = float(np.mean([m(y_true, raw_preds[s]) for s in range(S)]))
    metric_oe = float(np.mean([m(y_true, oe_preds[s]) for s in range(S)]))

    rng = np.random.RandomState(seed)
    boot_r = np.empty(n_boot)
    boot_d = np.empty(n_boot)
    for b in range(n_boot):
        gsel = rng.randint(0, n_groups, n_groups)             # resample GROUPS
        idx = np.concatenate([gmap[gkeys[g]] for g in gsel])
        boot_r[b] = _mean_seed_ratio(m, y_true, raw_preds, oe_preds, idx)
        boot_d[b] = _mean_seed_delta(m, y_true, raw_preds, oe_preds, idx)

    jack_r = jack_d = None
    if acceleration:
        all_idx = np.arange(N)
        jack_r = np.empty(n_groups)
        jack_d = np.empty(n_groups)
        for j in range(n_groups):
            keep = np.setdiff1d(all_idx, gmap[gkeys[j]], assume_unique=False)
            jack_r[j] = _mean_seed_ratio(m, y_true, raw_preds, oe_preds, keep)
            jack_d[j] = _mean_seed_delta(m, y_true, raw_preds, oe_preds, keep)

    ci_low, ci_high = _bca_interval(boot_r, theta_hat, jack_r)
    d_low, d_high = _bca_interval(boot_d, delta_hat, jack_d)
    return {
        "ratio": theta_hat,
        "metric_raw": metric_raw,
        "metric_oe": metric_oe,
        "between_seed_sd": between_seed_sd,
        "seed_ratios": [float(x) for x in seed_ratios],
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "delta": delta_hat,
        "delta_ci": [float(d_low), float(d_high)],
        "n_items": int(N),
        "n_seeds": int(S),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
    }
