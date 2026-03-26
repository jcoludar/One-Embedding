"""Golden rules for the rigorous benchmark framework.

Every rule is a function that either passes silently or raises AssertionError
with a clear message. Rules are called before any metric is computed.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Rule 4: MetricResult — mandatory CI wrapper
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Every metric MUST be wrapped in this. No bare floats allowed."""
    value: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    n: int
    seeds_mean: Optional[float] = None
    seeds_std: Optional[float] = None
    ci_method: str = "percentile"

    def __post_init__(self):
        if self.ci_lower is None or self.ci_upper is None:
            raise ValueError(
                "Rule 4 violation: MetricResult requires ci_lower and ci_upper. "
                "Use bootstrap_ci() to compute confidence intervals."
            )


# ---------------------------------------------------------------------------
# Rule 1: Fair Comparison
# ---------------------------------------------------------------------------

def check_protein_level_comparison(
    pooling_method: str,
    raw_pooling: str,
) -> None:
    """Assert raw and compressed use the same pooling method."""
    assert pooling_method == raw_pooling, (
        f"Rule 1 violation: pooling mismatch. "
        f"Compressed uses '{pooling_method}', raw uses '{raw_pooling}'. "
        f"Both must match for fair comparison."
    )


def check_per_residue_comparison(
    probe_type: str,
    raw_probe_type: str,
    hp_grid: dict,
    raw_hp_grid: dict,
) -> None:
    """Assert raw and compressed use the same probe and hyperparameter grid."""
    assert probe_type == raw_probe_type, (
        f"Rule 1 violation: probe mismatch. "
        f"Compressed uses '{probe_type}', raw uses '{raw_probe_type}'."
    )
    assert hp_grid == raw_hp_grid, (
        f"Rule 1 violation: hyperparameter grid mismatch. "
        f"Compressed: {hp_grid}, raw: {raw_hp_grid}."
    )


# ---------------------------------------------------------------------------
# Rule 2: No Train/Test Leakage
# ---------------------------------------------------------------------------

def check_no_leakage(train_ids: list, test_ids: list) -> None:
    """Assert zero overlap between train and test."""
    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, (
        f"Rule 2 violation: train/test leakage detected. "
        f"{len(overlap)} overlapping IDs: {list(overlap)[:5]}..."
    )


# ---------------------------------------------------------------------------
# Rule 6: Class Balance Reporting
# ---------------------------------------------------------------------------

def check_class_balance(
    labels: np.ndarray,
    threshold_ratio: float = 3.0,
    min_class_samples: int = 100,
) -> dict:
    """Check class balance and return diagnostics.

    Does not raise — returns a report dict. Callers decide what to do.
    """
    classes, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    min_count = counts.min()
    ratio = max_count / max(min_count, 1)
    small_classes = [
        str(c) for c, n in zip(classes, counts) if n < min_class_samples
    ]
    return {
        "imbalanced": ratio > threshold_ratio,
        "max_ratio": float(ratio),
        "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
        "small_classes": small_classes,
    }


# ---------------------------------------------------------------------------
# Rule 11/14: Cross-Dataset Consistency
# ---------------------------------------------------------------------------

def check_cross_dataset_consistency(
    results: dict[str, float],
    warn_pp: float = 3.0,
    block_pp: float = 5.0,
) -> dict:
    """Compare retention across datasets for the same task.

    Args:
        results: {dataset_name: retention_pct} e.g. {"CB513": 98.8, "TS115": 97.1}
        warn_pp: Warn if max divergence exceeds this (pp)
        block_pp: Block if max divergence exceeds this (pp)

    Returns:
        dict with max_divergence, status ("ok", "warn", "block"), pairs.
    """
    names = list(results.keys())
    values = list(results.values())
    if len(values) < 2:
        return {"max_divergence": 0.0, "status": "ok", "pairs": []}

    pairs = []
    max_div = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            div = abs(values[i] - values[j])
            pairs.append((names[i], names[j], div))
            max_div = max(max_div, div)

    if max_div > block_pp:
        status = "block"
    elif max_div > warn_pp:
        status = "warn"
    else:
        status = "ok"

    return {"max_divergence": max_div, "status": status, "pairs": pairs}
