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
from metrics.statistics import bootstrap_ci, multi_seed_summary, averaged_multi_seed, cluster_bootstrap_ci
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
# Per-protein prediction collector (for cluster bootstrap)
# ---------------------------------------------------------------------------

def _collect_per_protein_predictions(
    embeddings: dict,
    scores_dict: dict,
    protein_ids: list,
    probe,
    max_len: int,
    min_residues: int = 5,
) -> dict:
    """Collect per-protein (y_true, y_pred) arrays from a fitted probe.

    Unlike _per_protein_spearman which computes rho per protein, this
    returns the raw arrays needed for pooled metrics and cluster bootstrap.

    Args:
        embeddings: {pid: (L, D)}
        scores_dict: {pid: (L,) float array}
        protein_ids: IDs to score.
        probe: Fitted sklearn estimator with .predict().
        max_len: Maximum residues per protein.
        min_residues: Minimum valid (non-NaN) residues required to include protein.

    Returns:
        {pid: {"y_true": ndarray, "y_pred": ndarray}} — only proteins with
        >= min_residues valid residues.
    """
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
        results[pid] = {"y_true": y_p, "y_pred": preds}

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
    disorder_threshold: float = 8.0,
) -> dict:
    """Run disorder regression benchmark with pooled Spearman rho + AUC-ROC.

    Headline metric: POOLED residue-level Spearman rho (matching SETH,
    ODiNPred, ADOPT, UdonPred literature standard). CI via cluster bootstrap
    (resample proteins, recompute pooled rho) to respect the hierarchical
    data structure (Davison & Hinkley 1997, Ch. 2.4).

    Secondary metric: AUC-ROC on binary Z < threshold (CAID standard).

    Supplementary: per-protein averaged Spearman rho (for transparency).

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
        disorder_threshold: Z-score threshold for binary disorder (default: 8.0).
            Z < threshold = disordered (positive class).

    Returns:
        dict with:
            "pooled_spearman_rho": MetricResult — headline (cluster bootstrap CI).
            "auc_roc": MetricResult — AUC-ROC on binary Z<threshold (cluster bootstrap CI).
            "per_protein_spearman_rho": MetricResult — supplementary (averaged multi-seed).
            "spearman_rho": MetricResult — backward-compat alias for per_protein_spearman_rho.
            "best_alpha": float from median seed.
            "n_train_residues": int.
            "n_test_residues": int.
            "n_test_proteins": int.
            "disorder_threshold": float.
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

    # --- Phase 1: Train probes across seeds, collect per-protein predictions ---
    seed_probe_results = []
    seed_per_protein_preds = []  # list of {pid: {"y_true": arr, "y_pred": arr}}

    for seed in seeds:
        probe_result = train_regression_probe(
            X_train, y_train, X_test, y_test,
            alpha_grid=alpha_grid, cv_folds=cv_folds, seed=seed,
        )
        seed_probe_results.append(probe_result)

        # Collect per-protein predictions for this seed
        fitted_probe = _get_fitted_ridge(X_train, y_train, probe_result["best_alpha"])
        per_protein_preds = _collect_per_protein_predictions(
            embeddings, scores, test_ids, fitted_probe, max_len, min_residues=5
        )
        seed_per_protein_preds.append(per_protein_preds)

    # --- Phase 2: Average predictions across seeds per-protein per-residue ---
    # Use intersection of proteins present in all seeds
    common_pids = sorted(
        set.intersection(*[set(sp.keys()) for sp in seed_per_protein_preds])
    )
    n_test_proteins = len(common_pids)

    averaged_clusters = {}  # {pid: {"y_true": arr, "y_pred": arr}}
    for pid in common_pids:
        y_true = seed_per_protein_preds[0][pid]["y_true"]  # same across seeds
        # Average predicted Z-scores across seeds
        y_pred_stacked = np.stack(
            [seed_per_protein_preds[s][pid]["y_pred"] for s in range(len(seeds))],
            axis=0,
        )
        y_pred_avg = np.mean(y_pred_stacked, axis=0)
        averaged_clusters[pid] = {"y_true": y_true, "y_pred": y_pred_avg}

    # --- Phase 3: POOLED Spearman rho with cluster bootstrap CI (headline) ---
    def _pooled_spearman(cluster_data: list[dict]) -> float:
        all_true = np.concatenate([d["y_true"] for d in cluster_data])
        all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
        rho, _ = _spearmanr(all_true, all_pred)
        return float(rho) if not np.isnan(rho) else 0.0

    pooled_rho_result = cluster_bootstrap_ci(
        averaged_clusters, _pooled_spearman,
        n_bootstrap=n_bootstrap, seed=seeds[0],
    )

    # --- Phase 4: AUC-ROC on binary Z<threshold with cluster bootstrap CI ---
    def _pooled_auc_roc(cluster_data: list[dict]) -> float:
        all_true = np.concatenate([d["y_true"] for d in cluster_data])
        all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
        # Z < threshold = disordered (positive class)
        binary_labels = (all_true < disorder_threshold).astype(int)
        # Negate predicted Z-scores so higher = more disordered
        pred_disorder = -all_pred
        # AUC-ROC requires both classes present
        if len(np.unique(binary_labels)) < 2:
            return 0.5
        return float(roc_auc_score(binary_labels, pred_disorder))

    auc_result = cluster_bootstrap_ci(
        averaged_clusters, _pooled_auc_roc,
        n_bootstrap=n_bootstrap, seed=seeds[0],
    )

    # --- Phase 5: Per-protein Spearman rho (supplementary, backward compat) ---
    seed_per_protein_rho_dicts = []
    for sp_preds in seed_per_protein_preds:
        rho_dict = {}
        for pid, data in sp_preds.items():
            rho, _ = _spearmanr(data["y_true"], data["y_pred"])
            rho_dict[pid] = float(rho) if not np.isnan(rho) else 0.0
        seed_per_protein_rho_dicts.append(rho_dict)

    per_protein_rho_result = averaged_multi_seed(
        seed_per_protein_rho_dicts, n_bootstrap=n_bootstrap, seed=seeds[0]
    )

    # --- Select median seed for best_alpha reporting ---
    per_seed_pooled = []
    for sp_preds in seed_per_protein_preds:
        cluster_data = [sp_preds[pid] for pid in sorted(sp_preds.keys())]
        per_seed_pooled.append(_pooled_spearman(cluster_data))
    sorted_idx = np.argsort(per_seed_pooled)
    median_idx = int(sorted_idx[len(sorted_idx) // 2])
    median_probe = seed_probe_results[median_idx]

    return {
        "pooled_spearman_rho": pooled_rho_result,
        "auc_roc": auc_result,
        "per_protein_spearman_rho": per_protein_rho_result,
        "spearman_rho": per_protein_rho_result,  # backward compat alias
        "best_alpha": median_probe["best_alpha"],
        "n_train_residues": int(len(y_train)),
        "n_test_residues": int(len(y_test)),
        "n_test_proteins": n_test_proteins,
        "disorder_threshold": disorder_threshold,
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
