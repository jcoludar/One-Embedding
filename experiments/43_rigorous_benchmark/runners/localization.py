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

from rules import MetricResult, check_class_balance
from metrics.statistics import (
    averaged_multi_seed,
    cluster_bootstrap_ci,
    paired_bootstrap_retention,
    paired_cluster_bootstrap_retention,
)
from probes.linear import train_classification_probe


def parse_deeploc_fasta(fasta_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse DeepLoc FASTA to {uniprot_id: sequence} and {uniprot_id: location}.

    Header formats:
        Train:     >Q5I0E9 Cell.membrane-M
        Test:      >Q9H400 Cell.membrane-M test
        setHARD:   >Q12981 Endoplasmic.reticulum-U new_test_set

    Location is extracted as the part before the hyphen in the second field.
    The membrane type suffix (-M for membrane, -U for unknown/soluble) is dropped.
    """
    sequences = {}
    labels = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                parts = line[1:].split()
                current_id = parts[0]
                loc_field = parts[1] if len(parts) > 1 else "Unknown"
                location = loc_field.rsplit("-", 1)[0]
                labels[current_id] = location
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences, labels

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


def _to_protein_vector(
    x: np.ndarray,
    method: str = "mean",
    dct_k: int = 4,
) -> np.ndarray:
    """Convert residue-level or already-pooled embedding to one protein vector.

    Args:
        x: Either a residue-level matrix (L, D) or an already-pooled vector (D,).
        method: "mean", "dct_k4", "dct_k8", or "protein_vec".
        dct_k: Number of DCT coefficients for DCT-based methods.

    Returns:
        Protein-level vector.
    """
    x = np.asarray(x, dtype=np.float32)

    # Already a protein-level vector
    if x.ndim == 1:
        return x

    if method == "mean":
        return x.mean(axis=0).astype(np.float32)

    if method in {"dct_k4", "dct_k8"}:
        coeffs = dct(x, axis=0, type=2, norm="ortho")[:dct_k]
        return coeffs.flatten().astype(np.float32)

    raise ValueError(f"Unknown feature method: {method}")


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
    feature_method: str = "mean",
    dct_k: int = 4,
) -> dict:
    """Run localization on one embedding space using protein-level features + LogReg.

    Returns Q10 accuracy, macro/weighted F1 (with bootstrap CIs), per-class F1,
    and per-protein scores for paired retention vs another embedding space.
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

    train_vecs = np.array([_to_protein_vector(embeddings[pid], method=feature_method, dct_k=dct_k) for pid in train_ids])
    test_vecs = np.array([_to_protein_vector(embeddings[pid], method=feature_method, dct_k=dct_k) for pid in test_ids])
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
            C_grid=C_grid, cv_folds=cv_folds, seed=seed,
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
    weighted_f1 = cluster_bootstrap_ci(
        clusters, _weighted_f1_from_clusters, n_bootstrap=n_bootstrap, seed=seeds[0] + 1,
    )
    per_class_f1 = _per_class_f1_from_clusters(list(clusters.values()))

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
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "per_protein_scores": per_protein_scores,
        "per_protein_predictions": clusters,
        "class_balance": balance,
        "best_C": best_C,
    }


def run_localization_benchmark(
    raw_embeddings: dict[str, np.ndarray],
    comp_embeddings: dict[str, np.ndarray],
    train_labels: dict[str, str],
    test_labels: dict[str, str],
    test_name: str,
    C_grid: list[float],
    cv_folds: int,
    seeds: list[int],
    n_bootstrap: int,
) -> dict:
    """Run raw vs compressed localization (Phase C API).

    Mean-pools per-residue embeddings, trains LogReg on both spaces, reports
    Q10 / macro F1 / weighted F1 with bootstrap CIs and paired retention.
    """
    train_ids = [pid for pid in train_labels
                 if pid in raw_embeddings and pid in comp_embeddings]
    test_ids = [pid for pid in test_labels
                if pid in raw_embeddings and pid in comp_embeddings]

    if len(train_ids) < 10 or len(test_ids) < 5:
        print(f"  SKIP {test_name}: insufficient data "
              f"(train={len(train_ids)}, test={len(test_ids)})")
        return {"status": "skipped", "train_n": len(train_ids), "test_n": len(test_ids)}

    print(f"  {test_name}: train={len(train_ids)}, test={len(test_ids)}")
    y_test = np.array([test_labels[pid] for pid in test_ids])
    balance = check_class_balance(y_test)
    print(f"  Class balance (test): imbalanced={balance['imbalanced']}, "
          f"max_ratio={balance['max_ratio']:.1f}")
    if balance["small_classes"]:
        print(f"  Small classes (<100 samples): {balance['small_classes']}")

    raw_result = run_localization_probe_benchmark(
        raw_embeddings, train_labels, test_labels, test_name,
        C_grid=C_grid, cv_folds=cv_folds, seeds=seeds, n_bootstrap=n_bootstrap,
        train_ids=train_ids, test_ids=test_ids, quiet=True,
    )
    if raw_result.get("status") != "done":
        return raw_result

    comp_result = run_localization_probe_benchmark(
        comp_embeddings, train_labels, test_labels, test_name,
        C_grid=C_grid, cv_folds=cv_folds, seeds=seeds, n_bootstrap=n_bootstrap,
        train_ids=train_ids, test_ids=test_ids, quiet=True,
    )
    if comp_result.get("status") != "done":
        return comp_result

    retention_q10 = paired_bootstrap_retention(
        raw_scores=raw_result["per_protein_scores"],
        comp_scores=comp_result["per_protein_scores"],
        n_bootstrap=n_bootstrap,
        seed=42,
    )
    retention_macro_f1 = paired_cluster_bootstrap_retention(
        raw_clusters=raw_result["per_protein_predictions"],
        comp_clusters=comp_result["per_protein_predictions"],
        statistic_fn=macro_f1_from_clusters,
        n_bootstrap=n_bootstrap,
        seed=42,
    )

    return {
        "status": "done",
        "test_name": test_name,
        "train_n": raw_result["train_n"],
        "test_n": raw_result["test_n"],
        "raw_q10": raw_result["q10"],
        "comp_q10": comp_result["q10"],
        "raw_macro_f1": raw_result["macro_f1"],
        "comp_macro_f1": comp_result["macro_f1"],
        "raw_weighted_f1": raw_result["weighted_f1"],
        "comp_weighted_f1": comp_result["weighted_f1"],
        "raw_per_class_f1": raw_result["per_class_f1"],
        "comp_per_class_f1": comp_result["per_class_f1"],
        "retention": retention_q10,
        "retention_macro_f1": retention_macro_f1,
        "class_balance": raw_result["class_balance"],
        "best_C_raw": raw_result["best_C"],
        "best_C_comp": comp_result["best_C"],
    }
