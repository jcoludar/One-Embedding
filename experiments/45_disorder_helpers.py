"""Shared helpers for disorder information loss investigation (Exp 45).

Functions for loading data, applying individual codec stages, training
per-stage Ridge probes, and computing per-protein Spearman rho.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.one_embedding.preprocessing import (
    compute_corpus_stats,
    center_embeddings,
    all_but_the_top,
)
from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.evaluation.per_residue_tasks import load_chezod_seth

DATA = ROOT / "data"


# ── Data loading ──────────────────────────────────────────────────────


def load_h5_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load {protein_id: (L, D) float32} from flat H5.

    Args:
        path: Path to an HDF5 file where each key maps to a 2D dataset.

    Returns:
        Dict mapping protein ID strings to (L, D) float32 arrays.
    """
    embs = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            embs[key] = f[key][:].astype(np.float32)
    return embs


def load_chezod_data():
    """Load CheZOD embeddings, Z-scores, and train/test split.

    Returns:
        (embs, scores, train_ids, test_ids) where embs and scores are
        dicts keyed by protein ID, and train/test_ids are lists filtered
        to proteins available in both embs and scores.
    """
    data_dir = DATA / "per_residue_benchmarks"
    sequences, scores, train_ids, test_ids = load_chezod_seth(data_dir)
    embs = load_h5_embeddings(DATA / "residue_embeddings" / "prot_t5_xl_chezod.h5")
    # Filter to proteins with both embeddings and scores
    train_ids = [p for p in train_ids if p in embs and p in scores]
    test_ids = [p for p in test_ids if p in embs and p in scores]
    return embs, scores, train_ids, test_ids


def _load_trizod_disorder_scores(json_path: Path) -> dict[str, np.ndarray]:
    """Load TriZOD disorder z-scores from moderate.json.

    The file contains one JSON object per line. Each object has:
        "ID": protein identifier (e.g. "19347_1_1_1")
        "zscores": list of floats or nulls (per-residue z-scores)

    Null values are converted to NaN.

    Returns:
        {protein_id: (L,) float64 array with NaN for missing positions}
    """
    scores = {}
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj["ID"]
            zscores = obj["zscores"]
            arr = np.array(
                [float(v) if v is not None else np.nan for v in zscores],
                dtype=np.float64,
            )
            scores[pid] = arr
    return scores


def load_trizod_data():
    """Load TriZOD embeddings, Z-scores, and train/test split.

    Returns:
        (embs, scores, train_ids, test_ids) — same interface as
        load_chezod_data().
    """
    split_path = DATA / "benchmark_suite" / "splits" / "trizod_predefined.json"
    with open(split_path) as f:
        split = json.load(f)
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]

    embs = load_h5_embeddings(DATA / "residue_embeddings" / "prot_t5_xl_trizod.h5")

    # Load TriZOD disorder scores from moderate.json
    scores_path = DATA / "per_residue_benchmarks" / "TriZOD" / "moderate.json"
    scores = _load_trizod_disorder_scores(scores_path)

    # Filter to proteins with both embeddings and scores
    train_ids = [p for p in train_ids if p in embs and p in scores]
    test_ids = [p for p in test_ids if p in embs and p in scores]
    return embs, scores, train_ids, test_ids


# ── Pipeline stages (each applied independently) ─────────────────────


def apply_abtt(embs: dict, corpus_stats: dict, k: int = 3) -> dict:
    """Apply centering + top-k PC removal to all proteins.

    Args:
        embs: {protein_id: (L, D) float32} embeddings.
        corpus_stats: Dict from compute_corpus_stats with 'mean_vec'
            and 'top_pcs' keys.
        k: Number of top principal components to remove.

    Returns:
        Dict of {protein_id: (L, D) float32} with ABTT applied.
    """
    mean_vec = corpus_stats["mean_vec"]
    top_pcs = corpus_stats["top_pcs"][:k]
    result = {}
    for pid, emb in embs.items():
        centered = center_embeddings(emb, mean_vec)
        result[pid] = all_but_the_top(centered, top_pcs)
    return result


def apply_rp(embs: dict, d_out: int = 768, seed: int = 42) -> dict:
    """Apply random orthogonal projection to d_out dimensions.

    Uses the same math as the codec: QR decomposition of a random
    Gaussian matrix, scaled by sqrt(d_in / d_out) to preserve norms.

    Args:
        embs: {protein_id: (L, D) float32} embeddings.
        d_out: Target dimensionality after projection.
        seed: Random seed for reproducible projection matrix.

    Returns:
        Dict of {protein_id: (L, d_out) float32} projected embeddings.
    """
    sample_d = next(iter(embs.values())).shape[1]
    rng = np.random.RandomState(seed)
    R = rng.randn(sample_d, d_out).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    proj = Q * np.sqrt(sample_d / d_out)
    return {pid: (emb @ proj) for pid, emb in embs.items()}


def apply_pq(embs: dict, codec: OneEmbeddingCodec) -> dict:
    """Apply PQ encode + decode using a fitted codec's PQ model.

    This round-trips through product quantization to simulate the
    lossy compression: encode to centroid indices then decode back
    to float32 reconstructed embeddings.

    Args:
        embs: {protein_id: (L, D) float32} embeddings (must match
            codec's fitted dimensionality).
        codec: A fitted OneEmbeddingCodec with quantization='pq'.

    Returns:
        Dict of {protein_id: (L, D) float32} PQ-reconstructed embeddings.
    """
    from src.one_embedding.quantization import pq_encode, pq_decode

    result = {}
    for pid, emb in embs.items():
        codes = pq_encode(emb, codec._pq_model)
        result[pid] = pq_decode(codes, codec._pq_model)
    return result


def apply_fp16(embs: dict) -> dict:
    """Apply fp16 quantization (cast to float16 and back to float32).

    Args:
        embs: {protein_id: (L, D) float32} embeddings.

    Returns:
        Dict of {protein_id: (L, D) float32} after fp16 round-trip.
    """
    return {
        pid: emb.astype(np.float16).astype(np.float32)
        for pid, emb in embs.items()
    }


# ── Ridge probe training + per-protein evaluation ────────────────────


def train_ridge_probe(
    embs: dict,
    scores: dict,
    train_ids: list,
    max_len: int = 512,
    alpha_grid: list = None,
):
    """Train RidgeCV on stacked residue embeddings, return fitted model.

    Stacks all residues from training proteins (up to max_len per protein),
    filters NaN z-scores, and fits RidgeCV with 3-fold CV.

    Args:
        embs: {protein_id: (L, D) float32} embeddings.
        scores: {protein_id: (L,) float64} z-score arrays (may contain NaN).
        train_ids: List of protein IDs to use for training.
        max_len: Maximum residues per protein (truncate longer proteins).
        alpha_grid: Regularization strengths for RidgeCV. Defaults to
            [0.01, 0.1, 1.0, 10.0, 100.0].

    Returns:
        Fitted sklearn.linear_model.RidgeCV model.
    """
    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    X_parts, y_parts = [], []
    for pid in train_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))
        for i in range(n):
            if not np.isnan(lab[i]):
                X_parts.append(emb[i])
                y_parts.append(lab[i])

    X = np.stack(X_parts).astype(np.float32)
    y = np.array(y_parts, dtype=np.float64)
    model = RidgeCV(alphas=alpha_grid, cv=3)
    model.fit(X, y)
    return model


def per_protein_rho(
    embs: dict,
    scores: dict,
    protein_ids: list,
    probe,
    max_len: int = 512,
    min_residues: int = 5,
) -> dict[str, float]:
    """Compute per-protein Spearman rho using a fitted probe.

    For each protein, predicts z-scores from embeddings via the probe,
    then computes Spearman correlation between predicted and true scores.

    Args:
        embs: {protein_id: (L, D) float32} embeddings.
        scores: {protein_id: (L,) float64} z-score arrays (may contain NaN).
        protein_ids: List of protein IDs to evaluate.
        probe: Fitted sklearn model with .predict() method.
        max_len: Maximum residues per protein.
        min_residues: Skip proteins with fewer valid residues.

    Returns:
        {protein_id: spearman_rho} for proteins with enough residues.
    """
    results = {}
    for pid in protein_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
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
        results[pid] = float(rho) if not np.isnan(rho) else 0.0
    return results


def pooled_rho(
    embs: dict,
    scores: dict,
    protein_ids: list,
    probe,
    max_len: int = 512,
) -> float:
    """Compute pooled residue-level Spearman rho (headline metric).

    Stacks all valid residues from all proteins, predicts in batch per
    protein, then computes a single Spearman rho across all residues.
    This matches the SETH/ODiNPred/ADOPT/UdonPred evaluation standard.

    Args:
        embs: {protein_id: (L, D) float32} embeddings.
        scores: {protein_id: (L,) float64} z-score arrays (may contain NaN).
        protein_ids: List of protein IDs to evaluate.
        probe: Fitted sklearn model with .predict() method.
        max_len: Maximum residues per protein.

    Returns:
        Single Spearman rho value across all pooled residues.
    """
    all_true_arr, all_pred_arr = [], []
    for pid in protein_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))
        X_p, y_p = [], []
        for i in range(n):
            if not np.isnan(lab[i]):
                X_p.append(emb[i])
                y_p.append(lab[i])
        if len(X_p) == 0:
            continue
        X_p = np.stack(X_p).astype(np.float32)
        preds = probe.predict(X_p)
        all_true_arr.extend(y_p)
        all_pred_arr.extend(preds.tolist())

    rho, _ = spearmanr(all_true_arr, all_pred_arr)
    return float(rho) if not np.isnan(rho) else 0.0


def per_protein_predictions(
    embs: dict,
    scores: dict,
    protein_ids: list,
    probe,
    max_len: int = 512,
    min_residues: int = 5,
) -> dict:
    """Collect per-protein {y_true, y_pred} arrays from a fitted probe.

    Useful for downstream analysis: error maps, scatter plots,
    per-protein diagnostics.

    Args:
        embs: {protein_id: (L, D) float32} embeddings.
        scores: {protein_id: (L,) float64} z-score arrays (may contain NaN).
        protein_ids: List of protein IDs to evaluate.
        probe: Fitted sklearn model with .predict() method.
        max_len: Maximum residues per protein.
        min_residues: Skip proteins with fewer valid residues.

    Returns:
        {protein_id: {"y_true": ndarray, "y_pred": ndarray}} for each
        protein with enough residues.
    """
    results = {}
    for pid in protein_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
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


# ── Protein characterization ─────────────────────────────────────────


def characterize_protein(
    scores: dict, pid: str, threshold: float = 8.0
) -> dict:
    """Compute properties of a protein's disorder profile.

    Args:
        scores: {protein_id: (L,) float64} z-score arrays.
        pid: Protein ID to characterize.
        threshold: Z-score threshold for disorder (Z < threshold = disordered,
            CheZOD convention: Z < 8 is disordered).

    Returns:
        Dict with keys: pid, n_residues, frac_disordered, frac_ordered,
        z_mean, z_std, z_min, z_max, z_median, is_bimodal.
        If no valid residues, returns {pid, n_residues: 0}.
    """
    z = np.asarray(scores[pid], dtype=np.float64)
    valid = z[~np.isnan(z)]
    n = len(valid)
    if n == 0:
        return {"pid": pid, "n_residues": 0}

    n_disordered = int(np.sum(valid < threshold))
    return {
        "pid": pid,
        "n_residues": n,
        "frac_disordered": n_disordered / n,
        "frac_ordered": 1 - n_disordered / n,
        "z_mean": float(np.mean(valid)),
        "z_std": float(np.std(valid)),
        "z_min": float(np.min(valid)),
        "z_max": float(np.max(valid)),
        "z_median": float(np.median(valid)),
        "is_bimodal": bool(np.std(valid) > 4.0),  # rough heuristic
    }
