"""Protein-level localization benchmark runner (DeepLoc / Light Attention).

Converts per-residue embeddings to protein-level features, trains CV-tuned
LogReg probes over seeds, and returns MetricResult outputs with bootstrap CIs.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rules import check_class_balance
from metrics.statistics import averaged_multi_seed, cluster_bootstrap_ci
from probes.linear import train_classification_probe


def parse_deeploc_metadata_csv(csv_path: Path, max_length: int | None = None,) -> tuple[dict[str, str], dict[str, str]]:
    """Parse DeepLoc metadata CSV to train/test label dictionaries.

    Expected columns:
        ACC, localization, membrane_type, split, sequence, length

    Returns:
        train_labels: {ACC: localization}
        test_labels: {ACC: localization}
    """
    import csv

    train_labels = {}
    test_labels = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"ACC", "localization", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        for row in reader:
            acc = row["ACC"].strip()
            label = row["localization"].strip()
            split = row["split"].strip().lower()

            if max_length is not None:
                length = int(row.get("length", 0))
                if length > max_length:
                    continue

            if not acc or not label:
                continue

            if split == "train":
                train_labels[acc] = label
            elif split == "test":
                test_labels[acc] = label

    return train_labels, test_labels


def _majority_vote_predictions(
    seed_predictions: list[dict[str, str]],
) -> dict[str, str]:
    """Majority-vote class predictions across seeds per protein."""
    common = sorted(set.intersection(*[set(s.keys()) for s in seed_predictions]))
    voted = {}
    for pid in common:
        votes = [s[pid] for s in seed_predictions]
        voted[pid] = Counter(votes).most_common(1)[0][0]
    return voted


def _prediction_clusters(
    test_ids: list[str],
    y_test: np.ndarray,
    voted_predictions: dict[str, str],
) -> dict[str, dict]:
    return {
        pid: {"y_true": y_test[i], "y_pred": voted_predictions[pid]}
        for i, pid in enumerate(test_ids)
        if pid in voted_predictions
    }


def macro_f1_from_clusters(cluster_list: list[dict]) -> float:
    y_true = [c["y_true"] for c in cluster_list]
    y_pred = [c["y_pred"] for c in cluster_list]
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _weighted_f1_from_clusters(cluster_list: list[dict]) -> float:
    y_true = [c["y_true"] for c in cluster_list]
    y_pred = [c["y_pred"] for c in cluster_list]
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _per_class_f1_from_clusters(cluster_list: list[dict]) -> dict[str, float]:
    y_true = [c["y_true"] for c in cluster_list]
    y_pred = [c["y_pred"] for c in cluster_list]
    labels = sorted(set(y_true) | set(y_pred))
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {label: float(score) for label, score in zip(labels, scores)}


_DCT_K = {"dct_k4": 4, "dct_k8": 8}


def _to_protein_vector(x: np.ndarray, method: str = "mean") -> np.ndarray:
    """Convert a residue-level (or already-pooled) embedding to one protein vector.

    The number of DCT coefficients is derived from the method name ("dct_k4" -> 4,
    "dct_k8" -> 8), so a caller cannot accidentally request 8 coefficients while the
    pooling produces 4.

    Args:
        x: A residue-level matrix (L, D), or an already-pooled vector (V,). A 1-D
           input is assumed to have been produced by this module (e.g. via
           extract_protein_features_from_codec with the same method) and is returned
           unchanged.
        method: "mean", "dct_k4", or "dct_k8".

    Returns:
        A fixed-length protein-level vector: (D,) for mean, (k * D,) for DCT — the
        DCT length is constant across proteins even when L < k (zero-padded).
    """
    x = np.asarray(x, dtype=np.float32)

    # Already a protein-level vector — assumed pooled with the same method upstream.
    if x.ndim == 1:
        return x

    if method == "mean":
        return x.mean(axis=0).astype(np.float32)

    if method in _DCT_K:
        k = _DCT_K[method]
        coeffs = dct(x, axis=0, type=2, norm="ortho")  # (L, D)
        D = coeffs.shape[1]
        out = np.zeros((k, D), dtype=np.float32)
        rows = min(k, coeffs.shape[0])  # proteins shorter than k yield < k coeffs
        out[:rows] = coeffs[:rows]
        return out.flatten()

    raise ValueError(f"Unknown feature method: {method}")


def extract_protein_features_from_codec(
    raw_embeddings: dict[str, np.ndarray],
    codec,
    feature_method: str,
) -> dict[str, np.ndarray]:
    """Memory-safe codec feature extraction for protein-level tasks.

    Decodes one protein at a time, pools it to a single protein vector via
    ``_to_protein_vector`` (the same pooling used for the raw baseline, so raw and
    compressed features are constructed identically), and discards the decoded
    (L, D) residue matrix immediately. Only the protein vectors are retained, so
    peak memory is O(n_proteins * feature_dim) rather than O(sum of L * D).
    """
    vectors = {}
    n = len(raw_embeddings)
    for i, (pid, emb) in enumerate(raw_embeddings.items(), start=1):
        enc = codec.encode(emb)
        decoded = np.asarray(codec.decode_per_residue(enc), dtype=np.float32)
        vectors[pid] = _to_protein_vector(decoded, method=feature_method)
        del decoded
        if i % 1000 == 0:
            print(f"  processed {i}/{n} proteins")
    return vectors


def run_localization_probe_benchmark(
    embeddings: dict[str, np.ndarray],
    train_labels: dict[str, str],
    test_labels: dict[str, str],
    test_name: str,
    C_grid: list[float],
    cv_folds: int,
    seeds: list[int],
    n_bootstrap: int,
    train_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
    quiet: bool = False,
    
) -> dict:
    """Run localization on one embedding space using protein-level features + LogReg.

    Returns Q10 accuracy, macro/weighted F1 (with bootstrap CIs), per-class F1,
    and per-protein scores for paired retention vs another embedding space.

    Aggregation across seeds differs by metric (by design): Q10 is the per-protein
    accuracy averaged over seeds (each protein's 0/1 correctness is meaned, then
    bootstrapped), while the F1 scores are computed from the per-protein majority-vote
    prediction across seeds (a seed ensemble). The paired-retention pairing keys on
    protein id either way, so raw and compressed are always compared protein-for-protein.
    """
    if train_ids is None:
        train_ids = [pid for pid in train_labels if pid in embeddings]
    if test_ids is None:
        test_ids = [pid for pid in test_labels if pid in embeddings]

    if len(train_ids) < 10 or len(test_ids) < 5:
        if not quiet:
            print(f"  SKIP {test_name}: insufficient data "
                  f"(train={len(train_ids)}, test={len(test_ids)})")
        return {"status": "skipped", "train_n": len(train_ids), "test_n": len(test_ids)}

    if not quiet:
        print(f"  {test_name}: train={len(train_ids)}, test={len(test_ids)}")

    train_vecs = np.array([embeddings[pid] for pid in train_ids])
    test_vecs = np.array([embeddings[pid] for pid in test_ids])
    y_train = np.array([train_labels[pid] for pid in train_ids])
    y_test = np.array([test_labels[pid] for pid in test_ids])

    balance = check_class_balance(y_test)
    if not quiet:
        print(f"  Class balance (test): imbalanced={balance['imbalanced']}, "
              f"max_ratio={balance['max_ratio']:.1f}")
        if balance["small_classes"]:
            print(f"  Small classes (<100 samples): {balance['small_classes']}")

    seed_correctness = []
    seed_predictions = []
    best_C = None

    for seed in seeds:
        probe = train_classification_probe(
            X_train=train_vecs, y_train=y_train,
            X_test=test_vecs, y_test=y_test,
            C_grid=C_grid, cv_folds=cv_folds, seed=seed, n_jobs=4
        )
        best_C = probe["best_C"]
        seed_correctness.append({
            test_ids[i]: 1.0 if probe["predictions"][i] == y_test[i] else 0.0
            for i in range(len(test_ids))
        })
        seed_predictions.append({
            test_ids[i]: probe["predictions"][i]
            for i in range(len(test_ids))
        })

    q10 = averaged_multi_seed(seed_correctness, n_bootstrap=n_bootstrap, seed=seeds[0])

    voted = _majority_vote_predictions(seed_predictions)
    clusters = _prediction_clusters(test_ids, y_test, voted)

    macro_f1 = cluster_bootstrap_ci(
        clusters, macro_f1_from_clusters, n_bootstrap=n_bootstrap, seed=seeds[0],
    )
    

    common_pids = sorted(set.intersection(*[set(s.keys()) for s in seed_correctness]))
    per_protein_scores = {
        pid: float(np.mean([s[pid] for s in seed_correctness]))
        for pid in common_pids
    }

    return {
        "status": "done",
        "test_name": test_name,
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "q10": q10,
        "macro_f1": macro_f1,
        "per_protein_scores": per_protein_scores,
        "per_protein_predictions": clusters,
        "class_balance": balance,
        "best_C": best_C,
    }
