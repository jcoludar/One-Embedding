"""Per-residue benchmark runners: SS3, SS8, disorder.

Each runner enforces golden rules (no leakage, class balance reporting),
stacks per-protein embeddings into residue-level arrays, CV-tunes probes,
and returns MetricResult with bootstrap CI and multi-seed summary.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rules import MetricResult, check_no_leakage, check_class_balance
from metrics.statistics import bootstrap_ci, multi_seed_summary, averaged_multi_seed
from probes.linear import train_classification_probe, train_regression_probe


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

SS3_MAP = {"H": 0, "E": 1, "C": 2, "L": 2}
SS8_MAP = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "C": 7, "L": 7}


# ---------------------------------------------------------------------------
# Residue stacking
# ---------------------------------------------------------------------------

def _stack_residues(
    embeddings: dict,
    labels: dict,
    protein_ids: list,
    max_len: int = 512,
    label_map: dict = None,
) -> tuple:
    """Flatten per-protein data into per-residue arrays.

    For classification:
        - labels values are strings; chars are mapped via label_map.
        - Residues with chars not in label_map are skipped.

    For regression:
        - labels values are float arrays; NaN values are skipped.

    Args:
        embeddings: {protein_id: (L, D) array}
        labels: {protein_id: str or (L,) float array}
        protein_ids: Ordered list of protein IDs to include.
        max_len: Maximum residues to take from each protein (truncates to first max_len).
        label_map: Dict mapping label chars to int class indices (classification only).

    Returns:
        (X, y): X is (N, D) float32, y is (N,) int or float array.
    """
    X_parts = []
    y_parts = []

    for pid in protein_ids:
        emb = embeddings[pid][:max_len]   # (<=max_len, D)
        lab = labels[pid]

        if label_map is not None:
            # Classification: labels are strings
            if isinstance(lab, str):
                lab = lab[:max_len]
            else:
                lab = list(lab)[:max_len]

            for i, ch in enumerate(lab):
                if i >= len(emb):
                    break
                if ch in label_map:
                    X_parts.append(emb[i])
                    y_parts.append(label_map[ch])
        else:
            # Regression: labels are float arrays, skip NaN
            lab = np.asarray(lab, dtype=np.float64)[:max_len]
            n = min(len(emb), len(lab))
            for i in range(n):
                if not np.isnan(lab[i]):
                    X_parts.append(emb[i])
                    y_parts.append(lab[i])

    if len(X_parts) == 0:
        raise ValueError("No valid residues found after filtering.")

    X = np.stack(X_parts, axis=0).astype(np.float32)

    if label_map is not None:
        y = np.array(y_parts, dtype=np.int64)
    else:
        y = np.array(y_parts, dtype=np.float64)

    return X, y


# ---------------------------------------------------------------------------
# Per-protein Q3 scoring helper
# ---------------------------------------------------------------------------

def _per_protein_q3(
    embeddings: dict,
    labels: dict,
    protein_ids: list,
    probe,
    max_len: int,
    label_map: dict,
) -> dict:
    """Compute per-protein Q3 accuracy using a fitted probe.

    Args:
        embeddings: {pid: (L, D)}
        labels: {pid: str}
        protein_ids: IDs to score.
        probe: Fitted sklearn estimator with a .predict() method.
        max_len: Maximum residues per protein.
        label_map: Char-to-int map.

    Returns:
        {pid: float} — per-protein accuracy (only proteins with >= 1 valid residue).
    """
    scores = {}
    for pid in protein_ids:
        emb = embeddings[pid][:max_len]
        lab = labels[pid]

        if isinstance(lab, str):
            lab = lab[:max_len]
        else:
            lab = list(lab)[:max_len]

        X_p, y_p = [], []
        for i, ch in enumerate(lab):
            if i >= len(emb):
                break
            if ch in label_map:
                X_p.append(emb[i])
                y_p.append(label_map[ch])

        if len(X_p) == 0:
            continue

        X_p = np.stack(X_p).astype(np.float32)
        y_p = np.array(y_p, dtype=np.int64)
        preds = probe.predict(X_p)
        scores[pid] = float((preds == y_p).mean())

    return scores


# ---------------------------------------------------------------------------
# Per-protein Spearman rho helper
# ---------------------------------------------------------------------------

def _per_protein_spearman(
    embeddings: dict,
    scores_dict: dict,
    protein_ids: list,
    probe,
    max_len: int,
    min_residues: int = 5,
) -> dict:
    """Compute per-protein Spearman rho using a fitted probe.

    Args:
        embeddings: {pid: (L, D)}
        scores_dict: {pid: (L,) float array}
        protein_ids: IDs to score.
        probe: Fitted sklearn estimator with .predict().
        max_len: Maximum residues per protein.
        min_residues: Minimum valid (non-NaN) residues required to include protein.

    Returns:
        {pid: float} — per-protein Spearman rho.
    """
    from scipy.stats import spearmanr

    results = {}
    for pid in protein_ids:
        emb = embeddings[pid][:max_len]
        lab = np.asarray(scores_dict[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))

        X_p, y_p = [], []
        for i in range(n):
            if not np.isnan(lab[i]):
                X_p.append(emb[i])
                y_p.append(lab[i])

        if len(X_p) < min_residues:
            continue

        X_p = np.stack(X_p).astype(np.float32)
        y_p = np.array(y_p, dtype=np.float64)
        preds = probe.predict(X_p)
        rho, _ = spearmanr(y_p, preds)
        results[pid] = float(rho)

    return results


# ---------------------------------------------------------------------------
# SS3 benchmark
# ---------------------------------------------------------------------------

def run_ss3_benchmark(
    embeddings: dict,
    labels: dict,
    train_ids: list,
    test_ids: list,
    C_grid: list = None,
    cv_folds: int = 3,
    seeds: list = None,
    n_bootstrap: int = 10_000,
    max_len: int = 512,
) -> dict:
    """Run SS3 (Q3) per-residue classification benchmark.

    Args:
        embeddings: {protein_id: (L, D) float32 array}
        labels: {protein_id: str of DSSP characters}
        train_ids: Protein IDs for training (must not overlap test_ids).
        test_ids: Protein IDs for testing.
        C_grid: LogReg regularization grid. Default: [0.01, 0.1, 1.0, 10.0].
        cv_folds: Number of CV folds for C selection.
        seeds: List of random seeds. Default: [42, 43, 44].
        n_bootstrap: Bootstrap iterations for CI.
        max_len: Maximum residues per protein.

    Returns:
        dict with:
            "q3": MetricResult — headline Q3 with bootstrap CI (multi-seed).
            "per_class_acc": dict {class_name: float} from median seed.
            "class_balance": dict from check_class_balance on test labels.
            "best_C": best C from median seed.
            "n_train_residues": int.
            "n_test_residues": int.
    """
    if C_grid is None:
        C_grid = [0.01, 0.1, 1.0, 10.0]
    if seeds is None:
        seeds = [42, 43, 44]

    # Rule 2: no leakage
    check_no_leakage(train_ids, test_ids)

    # Stack residues
    X_train, y_train = _stack_residues(embeddings, labels, train_ids, max_len, SS3_MAP)
    X_test, y_test = _stack_residues(embeddings, labels, test_ids, max_len, SS3_MAP)

    # Rule 6: class balance
    class_balance = check_class_balance(y_test)

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


# ---------------------------------------------------------------------------
# SS8 benchmark
# ---------------------------------------------------------------------------

def run_ss8_benchmark(
    embeddings: dict,
    labels: dict,
    train_ids: list,
    test_ids: list,
    C_grid: list = None,
    cv_folds: int = 3,
    seeds: list = None,
    n_bootstrap: int = 10_000,
    max_len: int = 512,
) -> dict:
    """Run SS8 (Q8) per-residue classification benchmark.

    Same interface as run_ss3_benchmark but uses SS8_MAP (8 classes).

    Returns:
        dict with:
            "q8": MetricResult — headline Q8 with bootstrap CI (multi-seed).
            "per_class_acc": dict from median seed.
            "class_balance": dict from check_class_balance on test labels.
            "best_C": best C from median seed.
            "n_train_residues": int.
            "n_test_residues": int.
    """
    if C_grid is None:
        C_grid = [0.01, 0.1, 1.0, 10.0]
    if seeds is None:
        seeds = [42, 43, 44]

    check_no_leakage(train_ids, test_ids)

    X_train, y_train = _stack_residues(embeddings, labels, train_ids, max_len, SS8_MAP)
    X_test, y_test = _stack_residues(embeddings, labels, test_ids, max_len, SS8_MAP)

    class_balance = check_class_balance(y_test)

    seed_per_protein_scores = []  # list of {pid: q8} dicts
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
            embeddings, labels, test_ids, fitted_probe, max_len, SS8_MAP
        )
        seed_per_protein_scores.append(per_protein_scores)

    # Average across seeds, then bootstrap (Bouthillier et al. 2021)
    q8 = averaged_multi_seed(
        seed_per_protein_scores, n_bootstrap=n_bootstrap, seed=seeds[0]
    )

    # Select median-performing seed for per_class_acc and best_C reporting
    per_seed_means = [np.mean(list(s.values())) for s in seed_per_protein_scores]
    sorted_idx = np.argsort(per_seed_means)
    median_idx = int(sorted_idx[len(sorted_idx) // 2])
    median_probe = seed_probe_results[median_idx]

    return {
        "q8": q8,
        "per_class_acc": median_probe["per_class_acc"],
        "class_balance": class_balance,
        "best_C": median_probe["best_C"],
        "n_train_residues": int(len(y_train)),
        "n_test_residues": int(len(y_test)),
    }


# ---------------------------------------------------------------------------
# Disorder benchmark
# ---------------------------------------------------------------------------

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
) -> dict:
    """Run disorder (per-residue Spearman rho) regression benchmark.

    Args:
        embeddings: {protein_id: (L, D) float32 array}
        scores: {protein_id: (L,) float array with possible NaN}
        train_ids: Protein IDs for training.
        test_ids: Protein IDs for testing.
        alpha_grid: Ridge regularization grid. Default: [0.01, 0.1, 1.0, 10.0, 100.0].
        cv_folds: Number of CV folds.
        seeds: List of random seeds. Default: [42, 43, 44].
        n_bootstrap: Bootstrap iterations for CI.
        max_len: Maximum residues per protein.

    Returns:
        dict with:
            "spearman_rho": MetricResult — headline rho with bootstrap CI (multi-seed).
            "best_alpha": float from median seed.
            "n_train_residues": int.
            "n_test_residues": int.
    """
    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    if seeds is None:
        seeds = [42, 43, 44]

    check_no_leakage(train_ids, test_ids)

    X_train, y_train = _stack_residues(embeddings, scores, train_ids, max_len, label_map=None)
    X_test, y_test = _stack_residues(embeddings, scores, test_ids, max_len, label_map=None)

    seed_results = []
    seed_probe_results = []

    for seed in seeds:
        probe_result = train_regression_probe(
            X_train, y_train, X_test, y_test,
            alpha_grid=alpha_grid, cv_folds=cv_folds, seed=seed,
        )
        seed_probe_results.append(probe_result)

        # Per-protein Spearman rho for bootstrap CI
        fitted_probe = _get_fitted_ridge(X_train, y_train, probe_result["best_alpha"])
        per_protein_rhos = _per_protein_spearman(
            embeddings, scores, test_ids, fitted_probe, max_len, min_residues=5
        )

        if len(per_protein_rhos) == 0:
            # Fallback: use overall rho as single value
            per_protein_rhos = {"_all": probe_result["spearman_rho"]}

        metric_result = bootstrap_ci(per_protein_rhos, n_bootstrap=n_bootstrap, seed=seed)
        seed_results.append(metric_result)

    spearman_rho = multi_seed_summary(seed_results)

    values = [r.value for r in seed_results]
    sorted_idx = np.argsort(values)
    median_idx = int(sorted_idx[len(sorted_idx) // 2])
    median_probe = seed_probe_results[median_idx]

    # Also compute pooled residue-level rho (standard in literature, e.g. SETH)
    # This pools all test residues and computes one global Spearman rho.
    from scipy.stats import spearmanr as _spearmanr
    pooled_preds = median_probe["predictions"]
    pooled_rho, pooled_p = _spearmanr(y_test, pooled_preds)
    pooled_rho = float(pooled_rho) if not np.isnan(pooled_rho) else 0.0

    return {
        "spearman_rho": spearman_rho,
        "pooled_spearman_rho": pooled_rho,
        "best_alpha": median_probe["best_alpha"],
        "n_train_residues": int(len(y_train)),
        "n_test_residues": int(len(y_test)),
    }


# ---------------------------------------------------------------------------
# Internal helpers: refit probes for per-protein scoring
# ---------------------------------------------------------------------------

def _get_fitted_logreg(X_train, y_train, best_C, seed):
    """Refit LogisticRegression with the given C (no CV overhead)."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=best_C, max_iter=500, solver="lbfgs", random_state=seed)
    model.fit(X_train, y_train)
    return model


def _get_fitted_ridge(X_train, y_train, best_alpha):
    """Refit Ridge with the given alpha (no CV overhead)."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    return model
