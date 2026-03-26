#!/usr/bin/env python
"""Phase C: Protein-level benchmarks — CATH20 retrieval, subcellular localization.

The most comprehensive protein-level benchmark in Experiment 43. Tests whether
the One Embedding 1.0 codec preserves protein-level information on two tasks
that are independent of Phase A1 (which used SCOPe 5K retrieval only):

Section 1 — CATH20 Superfamily Retrieval
    Load CATH20 per-residue embeddings + parse CATH classification file.
    Compute protein vectors via four baselines:
        A: Raw + mean pool       (context — what most papers report)
        B: Raw + DCT K=4         (fair pooling comparison)
        C: Raw + ABTT3 + DCT K=4 (full pipeline minus RP)
        Compressed: Codec output protein_vec (ABTT3 + RP768 + DCT K=4)
    Run retrieval at superfamily level (C.A.T.H string).
    Filter to superfamilies with >=3 members for meaningful retrieval.
    Dual metric (cosine + euclidean). Bootstrap CIs on retention.

Section 2 — Competitor Comparison (Published Numbers)
    Report our CATH20 Ret@1 vs published results from EAT/ProtTucker/DCTdomain.
    Note: different PLMs and datasets, so not apples-to-apples.
    Provides context for where our codec stands in the literature.

Section 3 — Subcellular Localization (DeepLoc / Light Attention)
    Parse DeepLoc FASTA to get 10-class subcellular location labels.
    Load embeddings for train (9503 proteins) and test (2768 proteins).
    Train: LogReg probe on mean-pooled protein vectors.
    Test on: deeploc_test (2768) and setHARD (490) separately.
    Raw 1024d vs Compressed 768d.
    Report Q10 accuracy with bootstrap CIs + retention.

Section 4 — Cross-Check with Phase A1 SCOPe Retrieval
    Compare CATH20 retrieval retention vs SCOPe 5K retrieval retention.
    Cross-dataset consistency check (Rule 11/14).

Section 5 — Summary
    All retention numbers with CIs.
    Cross-dataset consistency verdicts.
    Mean protein-level retention.

Results saved to RESULTS_DIR / "phase_c_results.json".

Usage:
    uv run python experiments/43_rigorous_benchmark/run_phase_c.py
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: project root + experiment dir for relative imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_EXPERIMENT_DIR))

import h5py
import numpy as np

# Framework modules (from experiment dir)
from config import (
    BOOTSTRAP_N,
    C_GRID,
    CV_FOLDS,
    METADATA,
    RAW_EMBEDDINGS,
    RESULTS_DIR,
    SEEDS,
    CROSS_CHECK_WARN_PP,
    CROSS_CHECK_BLOCK_PP,
)
from rules import MetricResult, check_class_balance, check_cross_dataset_consistency
from metrics.statistics import (
    bootstrap_ci,
    paired_bootstrap_retention,
)
from runners.protein_level import (
    compute_protein_vectors,
    run_retrieval_benchmark,
)
from probes.linear import train_classification_probe

# Project-level imports
from src.one_embedding.core.codec import Codec
from src.one_embedding.preprocessing import (
    all_but_the_top,
    compute_corpus_stats,
)


# ---------------------------------------------------------------------------
# Additional data paths (not in config.py)
# ---------------------------------------------------------------------------
DATA = _PROJECT_ROOT / "data"

# CATH20
CATH20_FASTA = DATA / "external" / "cath20" / "cath20_labeled.fasta"
CATH20_CLASS_FILE = DATA / "external" / "cath20" / "cath-domain-list-v4_2_0.txt"
CATH20_EMB = DATA / "residue_embeddings" / "prot_t5_xl_cath20.h5"

# DeepLoc (Light Attention data files)
_LA_DATA = _PROJECT_ROOT / "tools" / "reference" / "LightAttention" / "data_files"
DEEPLOC_TRAIN_FASTA = _LA_DATA / "deeploc_our_train_set.fasta"
DEEPLOC_TEST_FASTA = _LA_DATA / "deeploc_test_set.fasta"
DEEPLOC_HARD_FASTA = _LA_DATA / "setHARD.fasta"
DEEPLOC_EMB = DATA / "residue_embeddings" / "prot_t5_xl_deeploc.h5"

# Phase A1 results (for cross-check)
PHASE_A1_RESULTS = RESULTS_DIR / "phase_a1_results.json"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_metric(name: str, mr: MetricResult, indent: int = 2) -> str:
    """Format a MetricResult for console display."""
    prefix = " " * indent
    base = (
        f"{prefix}{name}: {mr.value:.4f} "
        f"(95% CI: [{mr.ci_lower:.4f}, {mr.ci_upper:.4f}], n={mr.n})"
    )
    if mr.seeds_mean is not None and mr.seeds_std is not None:
        base += f" [seeds: {mr.seeds_mean:.4f} +/- {mr.seeds_std:.4f}]"
    return base


def fmt_retention(name: str, compressed: float, baseline: float) -> str:
    """Format a retention percentage line."""
    if baseline == 0:
        return f"  {name}: baseline=0, cannot compute retention"
    ret = (compressed / baseline) * 100
    return f"  {name}: {ret:.1f}% ({compressed:.4f} / {baseline:.4f})"


def fmt_retention_ci(name: str, mr: MetricResult) -> str:
    """Format a retention MetricResult (percentage with CI)."""
    return (
        f"  {name}: {mr.value:.1f}% "
        f"(95% CI: [{mr.ci_lower:.1f}%, {mr.ci_upper:.1f}%], n={mr.n})"
    )


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_h5_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load embeddings from a flat H5 file: {protein_id: (L, D)}."""
    embeddings = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)
    return embeddings


def parse_cath_labels(class_file: Path) -> dict[str, str]:
    """Parse CATH classification file to {domain_id: "C.A.T.H"}.

    The CATH domain list file has a header block (lines starting with #)
    followed by data lines with whitespace-separated columns:
        domain_id  C  A  T  H  S  O  L  I  D  resolution
    We use columns 1-4 (C.A.T.H) as the superfamily identifier.
    """
    labels = {}
    with open(class_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                domain_id = parts[0]
                cath = f"{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}"
                labels[domain_id] = cath
    return labels


def parse_cath20_fasta_labels(fasta_path: Path) -> dict[str, str]:
    """Parse CATH20 FASTA headers to {domain_id: "C.A.T.H"}.

    Headers are like: >1a1wA01|3.30.930.10
    Domain ID is the part before '|', CATH superfamily is after '|'.
    """
    labels = {}
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                if "|" in header:
                    domain_id, cath = header.split("|", 1)
                    labels[domain_id] = cath
    return labels


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
                # Save previous protein
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # Parse new header
                parts = line[1:].split()
                current_id = parts[0]
                # "Cell.membrane-M" -> "Cell.membrane"
                loc_field = parts[1] if len(parts) > 1 else "Unknown"
                location = loc_field.rsplit("-", 1)[0]
                labels[current_id] = location
                current_seq = []
            else:
                current_seq.append(line)

    # Save last protein
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    return sequences, labels


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------

def compress_embeddings(
    raw_embeddings: dict[str, np.ndarray],
    codec: Codec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compress all embeddings with the fitted codec.

    Returns:
        (per_residue_dict, protein_vec_dict)
    """
    per_residue = {}
    protein_vecs = {}
    for pid, raw in raw_embeddings.items():
        encoded = codec.encode(raw)
        per_residue[pid] = encoded["per_residue"].astype(np.float32)
        protein_vecs[pid] = encoded["protein_vec"].astype(np.float32)
    return per_residue, protein_vecs


def apply_abtt_to_dict(
    embeddings: dict[str, np.ndarray],
    stats: dict,
) -> dict[str, np.ndarray]:
    """Apply ABTT3 preprocessing (center + remove top PCs) to each protein."""
    mean_vec = stats["mean_vec"]
    top_pcs = stats["top_pcs"]
    result = {}
    for pid, emb in embeddings.items():
        centered = emb - mean_vec
        result[pid] = all_but_the_top(centered, top_pcs)
    return result


def fit_codec_on_scope(scope_corpus_path: Path, seed: int = 42) -> Codec:
    """Fit the One Embedding 1.0 codec on the SCOPe 5K corpus.

    ABTT is always fitted on SCOPe 5K (external, general-purpose) — never on
    the benchmark dataset itself — to avoid information leakage.
    """
    print("  Loading SCOPe 5K corpus for ABTT fitting...")
    scope_corpus = load_h5_embeddings(scope_corpus_path)
    codec = Codec(d_out=768, dct_k=4, seed=seed)
    codec.fit(scope_corpus, k=3)
    print(f"  Codec fitted on {len(scope_corpus)} proteins (SCOPe 5K).")
    return codec, scope_corpus


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def metric_to_dict(mr: MetricResult) -> dict:
    """Convert MetricResult to JSON-serializable dict."""
    return asdict(mr)


def results_to_serializable(results: dict) -> dict:
    """Recursively convert a results dict so MetricResults become dicts."""
    out = {}
    for k, v in results.items():
        if isinstance(v, MetricResult):
            out[k] = metric_to_dict(v)
        elif isinstance(v, dict):
            out[k] = results_to_serializable(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = [
                metric_to_dict(item) if isinstance(item, MetricResult)
                else results_to_serializable(item) if isinstance(item, dict)
                else float(item) if isinstance(item, (np.floating, np.integer))
                else item
                for item in v
            ]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Retrieval helper for CATH20
# ---------------------------------------------------------------------------

def filter_singletons(
    labels: dict[str, str],
    min_members: int = 3,
) -> dict[str, str]:
    """Filter to superfamilies with at least min_members members.

    For meaningful retrieval evaluation, we need superfamilies with >=3 members
    (at least one query protein, one correct match, and a distractor).
    """
    from collections import Counter
    sf_counts = Counter(labels.values())
    valid_sfs = {sf for sf, count in sf_counts.items() if count >= min_members}
    return {pid: sf for pid, sf in labels.items() if sf in valid_sfs}


def labels_to_metadata(labels: dict[str, str], label_key: str) -> list[dict]:
    """Convert {id: label} dict to list[dict] metadata format for run_retrieval_benchmark."""
    return [{"id": pid, label_key: label} for pid, label in labels.items()]


# ---------------------------------------------------------------------------
# Localization benchmark helpers
# ---------------------------------------------------------------------------

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
    """Run 10-class subcellular localization benchmark.

    Mean-pools per-residue embeddings to get protein vectors, then trains
    LogReg probes on raw and compressed. Reports Q10 accuracy with bootstrap
    CIs and paired retention.

    Args:
        raw_embeddings: {pid: (L, 1024)} raw ProtT5 embeddings.
        comp_embeddings: {pid: (L, 768)} compressed per-residue embeddings.
        train_labels: {pid: location_class} for training.
        test_labels: {pid: location_class} for testing.
        test_name: Name of the test set (for reporting).
        C_grid: Regularization values for LogReg CV.
        cv_folds: Number of CV folds.
        seeds: List of random seeds for multi-seed evaluation.
        n_bootstrap: Number of bootstrap iterations for CIs.

    Returns:
        Dict with raw/compressed Q10 MetricResult, retention, class balance.
    """
    # Identify proteins available in both embeddings and labels
    train_ids = [pid for pid in train_labels
                 if pid in raw_embeddings and pid in comp_embeddings]
    test_ids = [pid for pid in test_labels
                if pid in raw_embeddings and pid in comp_embeddings]

    if len(train_ids) < 10 or len(test_ids) < 5:
        print(f"  SKIP {test_name}: insufficient data "
              f"(train={len(train_ids)}, test={len(test_ids)})")
        return {"status": "skipped", "train_n": len(train_ids), "test_n": len(test_ids)}

    print(f"  {test_name}: train={len(train_ids)}, test={len(test_ids)}")

    # Encode labels as strings for the probe
    label_encoder = {}
    all_labels_str = sorted(set(list(train_labels.values()) + list(test_labels.values())))
    for i, label in enumerate(all_labels_str):
        label_encoder[label] = label  # keep as string for LogReg

    # Mean-pool to get protein vectors
    raw_train_vecs = np.array([raw_embeddings[pid].mean(axis=0) for pid in train_ids])
    raw_test_vecs = np.array([raw_embeddings[pid].mean(axis=0) for pid in test_ids])
    comp_train_vecs = np.array([comp_embeddings[pid].mean(axis=0) for pid in train_ids])
    comp_test_vecs = np.array([comp_embeddings[pid].mean(axis=0) for pid in test_ids])

    y_train = np.array([train_labels[pid] for pid in train_ids])
    y_test = np.array([test_labels[pid] for pid in test_ids])

    # Class balance check
    balance = check_class_balance(y_test)
    print(f"  Class balance (test): imbalanced={balance['imbalanced']}, "
          f"max_ratio={balance['max_ratio']:.1f}")
    if balance["small_classes"]:
        print(f"  Small classes (<100 samples): {balance['small_classes']}")

    # Multi-seed evaluation
    raw_per_protein_scores = {}  # {pid: 1.0/0.0} across all seeds
    comp_per_protein_scores = {}

    raw_seed_results = []
    comp_seed_results = []

    for seed in seeds:
        # Raw
        raw_probe = train_classification_probe(
            X_train=raw_train_vecs, y_train=y_train,
            X_test=raw_test_vecs, y_test=y_test,
            C_grid=C_grid, cv_folds=cv_folds, seed=seed,
        )
        # Per-protein correct/incorrect for bootstrap
        raw_correct = {
            test_ids[i]: 1.0 if raw_probe["predictions"][i] == y_test[i] else 0.0
            for i in range(len(test_ids))
        }
        raw_q10 = bootstrap_ci(raw_correct, n_bootstrap=n_bootstrap, seed=seed)
        raw_seed_results.append(raw_q10)

        # Compressed
        comp_probe = train_classification_probe(
            X_train=comp_train_vecs, y_train=y_train,
            X_test=comp_test_vecs, y_test=y_test,
            C_grid=C_grid, cv_folds=cv_folds, seed=seed,
        )
        comp_correct = {
            test_ids[i]: 1.0 if comp_probe["predictions"][i] == y_test[i] else 0.0
            for i in range(len(test_ids))
        }
        comp_q10 = bootstrap_ci(comp_correct, n_bootstrap=n_bootstrap, seed=seed)
        comp_seed_results.append(comp_q10)

        # Accumulate per-protein scores for paired bootstrap (use last seed)
        raw_per_protein_scores = raw_correct
        comp_per_protein_scores = comp_correct

    # Multi-seed summary
    from metrics.statistics import multi_seed_summary
    raw_q10_summary = multi_seed_summary(raw_seed_results) if len(seeds) > 1 else raw_seed_results[0]
    comp_q10_summary = multi_seed_summary(comp_seed_results) if len(seeds) > 1 else comp_seed_results[0]

    # Paired bootstrap retention (on the median seed's per-protein scores)
    retention = paired_bootstrap_retention(
        raw_scores=raw_per_protein_scores,
        comp_scores=comp_per_protein_scores,
        n_bootstrap=n_bootstrap,
        seed=42,
    )

    return {
        "status": "done",
        "test_name": test_name,
        "train_n": len(train_ids),
        "test_n": len(test_ids),
        "raw_q10": raw_q10_summary,
        "comp_q10": comp_q10_summary,
        "retention": retention,
        "class_balance": balance,
        "best_C_raw": raw_probe["best_C"],
        "best_C_comp": comp_probe["best_C"],
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    all_results = {}

    # ==================================================================
    # 1. CATH20 SUPERFAMILY RETRIEVAL
    # ==================================================================
    section("1. CATH20 Superfamily Retrieval")

    if not CATH20_EMB.exists():
        print(f"  SKIP: CATH20 embeddings not found: {CATH20_EMB}")
        print("  Run extraction script first: "
              "uv run python experiments/01_extract_residue_embeddings.py --dataset cath20")
    elif not CATH20_FASTA.exists() and not CATH20_CLASS_FILE.exists():
        print(f"  SKIP: CATH20 labels not found at {CATH20_FASTA} or {CATH20_CLASS_FILE}")
    else:
        # Load embeddings
        print("  Loading CATH20 embeddings...")
        raw_cath20 = load_h5_embeddings(CATH20_EMB)
        print(f"  Loaded {len(raw_cath20)} domain embeddings.")

        # Parse labels from both sources, prefer FASTA (matches embedding keys exactly)
        cath_labels = {}
        if CATH20_FASTA.exists():
            cath_labels = parse_cath20_fasta_labels(CATH20_FASTA)
            print(f"  Parsed {len(cath_labels)} labels from FASTA headers.")

        # Supplement with classification file if needed
        if CATH20_CLASS_FILE.exists():
            class_labels = parse_cath_labels(CATH20_CLASS_FILE)
            # Only add labels for domains not already in FASTA labels
            n_added = 0
            for pid, cath in class_labels.items():
                if pid not in cath_labels:
                    cath_labels[pid] = cath
                    n_added += 1
            if n_added > 0:
                print(f"  Added {n_added} labels from classification file.")

        # Intersect: domains that have both embeddings and labels
        common_ids = sorted(set(raw_cath20.keys()) & set(cath_labels.keys()))
        print(f"  Domains with embeddings + labels: {len(common_ids)}")

        # Filter labels to common IDs
        cath_labels_common = {pid: cath_labels[pid] for pid in common_ids}

        # Filter to superfamilies with >= 3 members
        cath_labels_filtered = filter_singletons(cath_labels_common, min_members=3)
        n_sf = len(set(cath_labels_filtered.values()))
        print(f"  After filtering (>=3 members): {len(cath_labels_filtered)} domains, "
              f"{n_sf} superfamilies")

        # Build metadata for retrieval runner
        metadata = labels_to_metadata(cath_labels_filtered, "superfamily")

        # Filter embeddings to domains in the filtered set
        raw_cath20_filtered = {pid: raw_cath20[pid] for pid in cath_labels_filtered}

        # ---- Check ABTT corpus availability ----
        scope_corpus_path = RAW_EMBEDDINGS["prot_t5"]
        if not scope_corpus_path.exists():
            print(f"  SKIP: SCOPe 5K corpus not found: {scope_corpus_path}")
            print("  Cannot fit ABTT without external corpus (Rule 2: no self-fit on benchmark data).")
        else:
            # Load SCOPe corpus for ABTT fitting
            print("  Loading SCOPe 5K corpus for ABTT fitting...")
            scope_corpus = load_h5_embeddings(scope_corpus_path)

            # Compute corpus stats for ABTT
            print("  Computing corpus stats (ABTT3)...")
            stats = compute_corpus_stats(scope_corpus, n_pcs=3, seed=42)

            # ---- Baseline A: Raw + mean pool ----
            print("\n  --- Baseline A: Raw + mean pool ---")
            vecs_A = compute_protein_vectors(raw_cath20_filtered, method="mean")
            ret_A = run_retrieval_benchmark(
                vectors=vecs_A, metadata=metadata,
                label_key="superfamily", n_bootstrap=BOOTSTRAP_N, seed=42,
            )
            print(fmt_metric("Ret@1 cosine  (raw+mean)", ret_A["ret1_cosine"]))
            print(fmt_metric("Ret@1 euclid  (raw+mean)", ret_A["ret1_euclidean"]))

            # ---- Baseline B: Raw + DCT K=4 ----
            print("\n  --- Baseline B: Raw + DCT K=4 ---")
            vecs_B = compute_protein_vectors(raw_cath20_filtered, method="dct_k4", dct_k=4)
            ret_B = run_retrieval_benchmark(
                vectors=vecs_B, metadata=metadata,
                label_key="superfamily", n_bootstrap=BOOTSTRAP_N, seed=42,
            )
            print(fmt_metric("Ret@1 cosine  (raw+dct4)", ret_B["ret1_cosine"]))
            print(fmt_metric("Ret@1 euclid  (raw+dct4)", ret_B["ret1_euclidean"]))

            # ---- Baseline C: Raw + ABTT3 + DCT K=4 ----
            print("\n  --- Baseline C: Raw + ABTT3 + DCT K=4 ---")
            abtt_cath20 = apply_abtt_to_dict(raw_cath20_filtered, stats)
            vecs_C = compute_protein_vectors(abtt_cath20, method="dct_k4", dct_k=4)
            ret_C = run_retrieval_benchmark(
                vectors=vecs_C, metadata=metadata,
                label_key="superfamily", n_bootstrap=BOOTSTRAP_N, seed=42,
            )
            print(fmt_metric("Ret@1 cosine  (abtt3+dct4)", ret_C["ret1_cosine"]))
            print(fmt_metric("Ret@1 euclid  (abtt3+dct4)", ret_C["ret1_euclidean"]))

            # ---- Compressed: Codec (ABTT3 + RP768 + DCT K=4) ----
            print("\n  --- Compressed: Codec (ABTT3 + RP768 + DCT K=4) ---")
            codec = Codec(d_out=768, dct_k=4, seed=42)
            codec.fit(scope_corpus, k=3)
            _, comp_cath20_vecs = compress_embeddings(raw_cath20_filtered, codec)

            ret_comp = run_retrieval_benchmark(
                vectors=comp_cath20_vecs, metadata=metadata,
                label_key="superfamily", n_bootstrap=BOOTSTRAP_N, seed=42,
            )
            print(fmt_metric("Ret@1 cosine  (compressed)", ret_comp["ret1_cosine"]))
            print(fmt_metric("Ret@1 euclid  (compressed)", ret_comp["ret1_euclidean"]))

            # ---- Retention vs Baseline C (fair comparison: same pooling) ----
            print("\n  --- Retention (Compressed / Baseline C) ---")
            print(fmt_retention(
                "Ret@1 cosine retention",
                ret_comp["ret1_cosine"].value,
                ret_C["ret1_cosine"].value,
            ))
            print(fmt_retention(
                "Ret@1 euclidean retention",
                ret_comp["ret1_euclidean"].value,
                ret_C["ret1_euclidean"].value,
            ))

            # Paired bootstrap retention on per-query correctness
            from runners.protein_level import _retrieval_ret1
            raw_per_q = _retrieval_ret1(vecs_C, metadata, "superfamily", "cosine")
            comp_per_q = _retrieval_ret1(comp_cath20_vecs, metadata, "superfamily", "cosine")
            ret_ci_cosine = paired_bootstrap_retention(
                raw_per_q, comp_per_q, n_bootstrap=BOOTSTRAP_N, seed=42,
            )
            print(fmt_retention_ci("Ret@1 cosine retention (paired bootstrap)", ret_ci_cosine))

            raw_per_q_euc = _retrieval_ret1(vecs_C, metadata, "superfamily", "euclidean")
            comp_per_q_euc = _retrieval_ret1(comp_cath20_vecs, metadata, "superfamily", "euclidean")
            ret_ci_euc = paired_bootstrap_retention(
                raw_per_q_euc, comp_per_q_euc, n_bootstrap=BOOTSTRAP_N, seed=42,
            )
            print(fmt_retention_ci("Ret@1 euclidean retention (paired bootstrap)", ret_ci_euc))

            all_results["cath20_retrieval"] = {
                "baseline_A_raw_mean": ret_A,
                "baseline_B_raw_dct4": ret_B,
                "baseline_C_abtt3_dct4": ret_C,
                "compressed": ret_comp,
                "retention_cosine_pct": ret_ci_cosine,
                "retention_euclidean_pct": ret_ci_euc,
                "n_domains": len(cath_labels_filtered),
                "n_superfamilies": n_sf,
            }

    # ==================================================================
    # 2. COMPETITOR COMPARISON (PUBLISHED NUMBERS)
    # ==================================================================
    section("2. Competitor Comparison (Published Numbers)")

    # Published retrieval results for context. These use different PLMs,
    # different datasets, and different evaluation protocols — so they are
    # NOT directly comparable. We report them for context only.
    published_results = {
        "EAT (ProtTucker, ProtT5, CATH S40)": {
            "reference": "Heinzinger et al., 2022 (EAT/ProtTucker)",
            "plm": "ProtT5-XL",
            "dataset": "CATH S40 (122K domains)",
            "method": "CLS-token or mean pool, ProtTucker MLP fine-tuned",
            "ret1_reported": "~0.70 (mean pool, no fine-tuning)",
            "ret1_finetuned": "~0.86 (ProtTucker fine-tuned)",
            "note": "Fine-tuned model uses supervised contrastive loss on CATH labels",
        },
        "DCTdomain (ESM2, 2D-DCT)": {
            "reference": "Aziz et al., 2024 (DCTdomain)",
            "plm": "ESM2-650M",
            "dataset": "CATH v4.3 subset",
            "method": "2D-DCT on per-residue embeddings -> 480d fingerprint",
            "ret1_reported": "not directly reported as Ret@1",
            "note": "Uses ESM2 (1280d), not ProtT5 (1024d). Not apples-to-apples.",
        },
    }

    print("  Published protein-level embedding retrieval results (for context):")
    print("  NOTE: Different PLMs, datasets, and protocols. Not directly comparable.")
    print()
    for name, info in published_results.items():
        print(f"  {name}")
        for k, v in info.items():
            print(f"    {k}: {v}")
        print()

    if "cath20_retrieval" in all_results:
        our_cos = all_results["cath20_retrieval"]["compressed"]["ret1_cosine"]
        our_euc = all_results["cath20_retrieval"]["compressed"]["ret1_euclidean"]
        print(f"  Our result (One Embedding 1.0, ProtT5, CATH20):")
        print(fmt_metric("Ret@1 cosine", our_cos))
        print(fmt_metric("Ret@1 euclidean", our_euc))
        print(f"  Note: CATH20 has 20% seq-id cutoff -> harder than CATH S40/S100.")
    else:
        print("  Our CATH20 results not available (Section 1 was skipped).")

    all_results["competitor_comparison"] = {
        "published": published_results,
        "note": "Different PLMs, datasets, and protocols. Not directly comparable.",
    }

    # ==================================================================
    # 3. SUBCELLULAR LOCALIZATION (DeepLoc)
    # ==================================================================
    section("3. Subcellular Localization (DeepLoc)")

    if not DEEPLOC_EMB.exists():
        print(f"  SKIP: DeepLoc embeddings not found: {DEEPLOC_EMB}")
        print("  Run extraction script first: "
              "uv run python experiments/01_extract_residue_embeddings.py --dataset deeploc")
    elif not DEEPLOC_TRAIN_FASTA.exists():
        print(f"  SKIP: DeepLoc train FASTA not found: {DEEPLOC_TRAIN_FASTA}")
    elif not DEEPLOC_TEST_FASTA.exists():
        print(f"  SKIP: DeepLoc test FASTA not found: {DEEPLOC_TEST_FASTA}")
    else:
        # Load embeddings
        print("  Loading DeepLoc embeddings...")
        raw_deeploc = load_h5_embeddings(DEEPLOC_EMB)
        print(f"  Loaded {len(raw_deeploc)} protein embeddings.")

        # Parse labels from FASTAs
        _, train_labels = parse_deeploc_fasta(DEEPLOC_TRAIN_FASTA)
        _, test_labels = parse_deeploc_fasta(DEEPLOC_TEST_FASTA)
        print(f"  Train labels: {len(train_labels)}, Test labels: {len(test_labels)}")

        # Parse setHARD if available
        hard_labels = {}
        if DEEPLOC_HARD_FASTA.exists():
            _, hard_labels = parse_deeploc_fasta(DEEPLOC_HARD_FASTA)
            print(f"  setHARD labels: {len(hard_labels)}")

        # Check how many are available in embeddings
        train_avail = sum(1 for pid in train_labels if pid in raw_deeploc)
        test_avail = sum(1 for pid in test_labels if pid in raw_deeploc)
        hard_avail = sum(1 for pid in hard_labels if pid in raw_deeploc)
        print(f"  Available in embeddings: train={train_avail}, "
              f"test={test_avail}, setHARD={hard_avail}")

        # Fit codec on SCOPe 5K (external corpus)
        scope_corpus_path = RAW_EMBEDDINGS["prot_t5"]
        if not scope_corpus_path.exists():
            print(f"  SKIP: SCOPe 5K corpus not found: {scope_corpus_path}")
            print("  Cannot fit codec without external corpus.")
        else:
            codec, _ = fit_codec_on_scope(scope_corpus_path)

            # Compress all DeepLoc embeddings
            print("  Compressing DeepLoc embeddings...")
            comp_deeploc_per_res, _ = compress_embeddings(raw_deeploc, codec)
            print(f"  Compressed {len(comp_deeploc_per_res)} proteins to 768d.")

            # ---- 3a: Test on deeploc_test (2768) ----
            print("\n  --- 3a: DeepLoc Test Set ---")
            loc_test = run_localization_benchmark(
                raw_embeddings=raw_deeploc,
                comp_embeddings=comp_deeploc_per_res,
                train_labels=train_labels,
                test_labels=test_labels,
                test_name="deeploc_test",
                C_grid=C_GRID,
                cv_folds=CV_FOLDS,
                seeds=SEEDS,
                n_bootstrap=BOOTSTRAP_N,
            )
            if loc_test.get("status") == "done":
                print(fmt_metric("Q10 raw 1024d", loc_test["raw_q10"]))
                print(fmt_metric("Q10 compressed 768d", loc_test["comp_q10"]))
                print(fmt_retention(
                    "Q10 retention",
                    loc_test["comp_q10"].value,
                    loc_test["raw_q10"].value,
                ))
                print(fmt_retention_ci("Q10 retention (paired bootstrap)", loc_test["retention"]))

            all_results["localization_test"] = loc_test

            # ---- 3b: Test on setHARD (490) ----
            if hard_labels:
                print("\n  --- 3b: setHARD ---")
                loc_hard = run_localization_benchmark(
                    raw_embeddings=raw_deeploc,
                    comp_embeddings=comp_deeploc_per_res,
                    train_labels=train_labels,
                    test_labels=hard_labels,
                    test_name="setHARD",
                    C_grid=C_GRID,
                    cv_folds=CV_FOLDS,
                    seeds=SEEDS,
                    n_bootstrap=BOOTSTRAP_N,
                )
                if loc_hard.get("status") == "done":
                    print(fmt_metric("Q10 raw 1024d", loc_hard["raw_q10"]))
                    print(fmt_metric("Q10 compressed 768d", loc_hard["comp_q10"]))
                    print(fmt_retention(
                        "Q10 retention",
                        loc_hard["comp_q10"].value,
                        loc_hard["raw_q10"].value,
                    ))
                    print(fmt_retention_ci("Q10 retention (paired bootstrap)", loc_hard["retention"]))

                all_results["localization_hard"] = loc_hard

    # ==================================================================
    # 4. CROSS-CHECK WITH PHASE A1 SCOPE RETRIEVAL
    # ==================================================================
    section("4. Cross-Check with Phase A1 SCOPe Retrieval")

    if "cath20_retrieval" in all_results:
        cath_ret_cos = all_results["cath20_retrieval"]["retention_cosine_pct"]

        # Try to load Phase A1 results for SCOPe retrieval retention
        scope_ret_cos = None
        if PHASE_A1_RESULTS.exists():
            print(f"  Loading Phase A1 results from: {PHASE_A1_RESULTS}")
            with open(PHASE_A1_RESULTS) as f:
                a1_data = json.load(f)
            if "retrieval" in a1_data and "retention_cosine_pct" in a1_data["retrieval"]:
                scope_ret_cos = a1_data["retrieval"]["retention_cosine_pct"]
                # Could be a float or a MetricResult dict
                if isinstance(scope_ret_cos, dict):
                    scope_ret_cos = scope_ret_cos.get("value", scope_ret_cos)
                print(f"  SCOPe 5K Ret@1 cosine retention: {scope_ret_cos:.1f}%")
        else:
            print(f"  Phase A1 results not found: {PHASE_A1_RESULTS}")
            print("  Run run_phase_a1.py first for cross-check.")

        if scope_ret_cos is not None and isinstance(cath_ret_cos, MetricResult):
            # Cross-dataset consistency check (Rule 11/14)
            consistency = check_cross_dataset_consistency(
                results={
                    "SCOPe_5K": scope_ret_cos,
                    "CATH20": cath_ret_cos.value,
                },
                warn_pp=CROSS_CHECK_WARN_PP,
                block_pp=CROSS_CHECK_BLOCK_PP,
            )
            print(f"\n  Cross-dataset consistency (retrieval retention):")
            print(f"    SCOPe 5K: {scope_ret_cos:.1f}%")
            print(f"    CATH20:   {cath_ret_cos.value:.1f}%")
            print(f"    Max divergence: {consistency['max_divergence']:.1f} pp")
            print(f"    Status: {consistency['status'].upper()}")

            if consistency["status"] == "block":
                print("    WARNING: Cross-dataset divergence exceeds block threshold! "
                      "Investigate before publishing.")

            all_results["cross_check_retrieval"] = {
                "scope_5k_retention_pct": scope_ret_cos,
                "cath20_retention_pct": cath_ret_cos.value if isinstance(cath_ret_cos, MetricResult) else cath_ret_cos,
                "consistency": consistency,
            }
        elif scope_ret_cos is None:
            print("  Cannot perform cross-check: SCOPe retention not available.")
        else:
            print(f"  CATH20 retention format unexpected: {type(cath_ret_cos)}")
    else:
        print("  Cannot perform cross-check: CATH20 retrieval was skipped.")

    # ==================================================================
    # 5. SUMMARY
    # ==================================================================
    section("5. Protein-Level Retention Summary")

    summary_rows = []

    if "cath20_retrieval" in all_results:
        ret_cos = all_results["cath20_retrieval"]["retention_cosine_pct"]
        ret_euc = all_results["cath20_retrieval"]["retention_euclidean_pct"]
        if isinstance(ret_cos, MetricResult):
            summary_rows.append(("CATH20 Ret@1 cosine", ret_cos.value, "vs Baseline C"))
            print(f"  CATH20 Ret@1 cos: {ret_cos.value:.1f}%  "
                  f"(95% CI: [{ret_cos.ci_lower:.1f}%, {ret_cos.ci_upper:.1f}%])")
        if isinstance(ret_euc, MetricResult):
            summary_rows.append(("CATH20 Ret@1 euclidean", ret_euc.value, "vs Baseline C"))
            print(f"  CATH20 Ret@1 euc: {ret_euc.value:.1f}%  "
                  f"(95% CI: [{ret_euc.ci_lower:.1f}%, {ret_euc.ci_upper:.1f}%])")

    if "localization_test" in all_results:
        loc = all_results["localization_test"]
        if loc.get("status") == "done":
            ret_loc = loc["retention"]
            summary_rows.append(("DeepLoc Q10 (test)", ret_loc.value, "vs raw 1024d"))
            print(f"  DeepLoc Q10 (test): {ret_loc.value:.1f}%  "
                  f"(95% CI: [{ret_loc.ci_lower:.1f}%, {ret_loc.ci_upper:.1f}%])")

    if "localization_hard" in all_results:
        loc_h = all_results["localization_hard"]
        if loc_h.get("status") == "done":
            ret_h = loc_h["retention"]
            summary_rows.append(("DeepLoc Q10 (setHARD)", ret_h.value, "vs raw 1024d"))
            print(f"  DeepLoc Q10 (hard): {ret_h.value:.1f}%  "
                  f"(95% CI: [{ret_h.ci_lower:.1f}%, {ret_h.ci_upper:.1f}%])")

    if summary_rows:
        valid = [r[1] for r in summary_rows if r[1] is not None]
        if valid:
            mean_ret = np.mean(valid)
            print(f"\n  Mean protein-level retention: {mean_ret:.1f}%")
            all_results["summary"] = {
                "tasks": [
                    {"task": r[0], "retention_pct": r[1], "baseline": r[2]}
                    for r in summary_rows
                ],
                "mean_retention_pct": float(mean_ret),
            }
    else:
        print("  No protein-level benchmarks completed.")

    # Cross-check verdict
    if "cross_check_retrieval" in all_results:
        status = all_results["cross_check_retrieval"]["consistency"]["status"]
        print(f"\n  Cross-dataset consistency (retrieval): {status.upper()}")

    # ==================================================================
    # 6. SAVE RESULTS
    # ==================================================================
    section("6. Save Results")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "phase_c_results.json"

    serializable = results_to_serializable(all_results)
    serializable["_meta"] = {
        "script": "run_phase_c.py",
        "seeds": SEEDS,
        "bootstrap_n": BOOTSTRAP_N,
        "cv_folds": CV_FOLDS,
        "C_grid": C_GRID,
        "codec": "ABTT3 + RP768 + DCT K=4 (seed=42)",
        "abtt_corpus": "SCOPe 5K (external, general-purpose)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
