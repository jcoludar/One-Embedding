#!/usr/bin/env python
"""Phase B: Cross-validation on independent datasets + multi-PLM (ESM2).

Validates One Embedding 1.0 codec (ABTT3 + RP768 + DCT K=4) retention across
independent test sets and a second PLM:

Section 1 — SS3/SS8 Cross-Validation
    Train probes on CB513 train set, test on: CB513 test, TS115, CASP12.
    Raw ProtT5 1024d vs Compressed 768d for each test set.
    Cross-dataset consistency check on retention values.

Section 2 — Disorder Cross-Validation
    Train on CheZOD1174, test on: CheZOD117, TriZOD348.
    Raw ProtT5 1024d vs Compressed 768d for each test set.
    Cross-dataset consistency check on retention values.

Section 3 — ESM2 Multi-PLM Validation
    SS3/SS8 on ESM2 CB513 (1280d -> 768d).
    Retrieval on ESM2 SCOPe 5K.
    Compare retention: ProtT5 vs ESM2.

Section 4 — Summary
    All retention numbers with CIs.
    Cross-dataset consistency verdicts.
    Multi-PLM agreement.

Results saved to RESULTS_DIR / "phase_b_results.json".

Usage:
    uv run python experiments/43_rigorous_benchmark/run_phase_b.py
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
import pandas as pd

# Framework modules (from experiment dir)
from config import (
    ALPHA_GRID,
    BOOTSTRAP_N,
    C_GRID,
    CV_FOLDS,
    LABELS,
    METADATA,
    RAW_EMBEDDINGS,
    RESULTS_DIR,
    SEEDS,
    SPLITS,
    CROSS_CHECK_WARN_PP,
    CROSS_CHECK_BLOCK_PP,
)
from rules import MetricResult, check_cross_dataset_consistency

# Runners (from experiment dir)
from runners.per_residue import (
    run_disorder_benchmark,
    run_ss3_benchmark,
    run_ss8_benchmark,
    _get_fitted_logreg,
    _get_fitted_ridge,
    _per_protein_q3,
    _per_protein_spearman,
    _stack_residues,
    SS3_MAP,
    SS8_MAP,
)
from runners.protein_level import (
    compute_protein_vectors,
    run_retrieval_benchmark,
)
from metrics.statistics import (
    bootstrap_ci,
    paired_bootstrap_retention,
    paired_cluster_bootstrap_retention,
    multi_seed_summary,
)

# Dataset loaders (from experiment dir)
from datasets.netsurfp import load_netsurfp_csv
from datasets.trizod import load_trizod_embeddings

# Project-level imports
from src.one_embedding.core.codec import Codec
from src.one_embedding.preprocessing import (
    all_but_the_top,
    compute_corpus_stats,
)
from src.evaluation.per_residue_tasks import load_cb513_csv, load_chezod_seth

# ---------------------------------------------------------------------------
# Additional data paths (not in config.py)
# ---------------------------------------------------------------------------
DATA = _PROJECT_ROOT / "data"

ESM2_CB513 = DATA / "residue_embeddings" / "esm2_650m_cb513.h5"
ESM2_SCOPE = DATA / "residue_embeddings" / "esm2_650m_medium5k.h5"
TS115_EMB = DATA / "residue_embeddings" / "prot_t5_xl_ts115.h5"
CASP12_EMB = DATA / "residue_embeddings" / "prot_t5_xl_casp12.h5"
TS115_CSV = DATA / "per_residue_benchmarks" / "TS115.csv"
CASP12_CSV = DATA / "per_residue_benchmarks" / "CASP12.csv"
TRIZOD_EMB = RAW_EMBEDDINGS.get("prot_t5_trizod", DATA / "residue_embeddings" / "prot_t5_xl_trizod.h5")
TRIZOD_SPLIT = SPLITS.get("trizod", DATA / "benchmark_suite" / "splits" / "trizod_predefined.json")
TRIZOD_SCORES_JSON = DATA / "per_residue_benchmarks" / "TriZOD" / "moderate.json"


# ---------------------------------------------------------------------------
# Display helpers (same as Phase A1)
# ---------------------------------------------------------------------------

def fmt_metric(name: str, mr: MetricResult, indent: int = 2) -> str:
    prefix = " " * indent
    base = (
        f"{prefix}{name}: {mr.value:.4f} "
        f"(95% CI: [{mr.ci_lower:.4f}, {mr.ci_upper:.4f}], n={mr.n})"
    )
    if mr.seeds_mean is not None and mr.seeds_std is not None:
        base += f" [seeds: {mr.seeds_mean:.4f} +/- {mr.seeds_std:.4f}]"
    return base


def fmt_retention(name: str, compressed: float, baseline: float) -> str:
    if baseline == 0:
        return f"  {name}: baseline=0, cannot compute retention"
    ret = (compressed / baseline) * 100
    return f"  {name}: {ret:.1f}% ({compressed:.4f} / {baseline:.4f})"


def fmt_retention_ci(name: str, mr: MetricResult) -> str:
    return (
        f"  {name}: {mr.value:.1f}% "
        f"(95% CI: [{mr.ci_lower:.1f}%, {mr.ci_upper:.1f}%], n={mr.n})"
    )


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Data loading helpers (same as Phase A1)
# ---------------------------------------------------------------------------

def load_h5_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load embeddings from a flat H5 file: {protein_id: (L, D)}."""
    embeddings = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key], dtype=np.float32)
    return embeddings


def load_split(path: Path) -> tuple[list[str], list[str]]:
    """Load train/test split from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["train_ids"], data["test_ids"]


def load_scope_metadata(path: Path) -> list[dict]:
    """Load SCOPe metadata CSV as list of dicts."""
    df = pd.read_csv(path)
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# TriZOD disorder score loader
# ---------------------------------------------------------------------------

def load_trizod_disorder_scores(json_path: Path) -> dict[str, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Codec helpers (same as Phase A1)
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
    """Apply ABTT3 preprocessing to each protein."""
    mean_vec = stats["mean_vec"]
    top_pcs = stats["top_pcs"]
    result = {}
    for pid, emb in embeddings.items():
        centered = emb - mean_vec
        result[pid] = all_but_the_top(centered, top_pcs)
    return result


# ---------------------------------------------------------------------------
# Serialization helpers (same as Phase A1)
# ---------------------------------------------------------------------------

def metric_to_dict(mr: MetricResult) -> dict:
    return asdict(mr)


_SKIP_KEYS = {"per_protein_scores", "per_protein_predictions", "per_query_cosine", "per_query_euclidean"}


def results_to_serializable(results: dict) -> dict:
    out = {}
    for k, v in results.items():
        if k in _SKIP_KEYS:
            continue
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
# Cross-dataset SS3/SS8 helper
# ---------------------------------------------------------------------------

def run_cross_dataset_ss_benchmark(
    *,
    task: str,
    label_map: dict,
    metric_key: str,
    cb513_raw: dict[str, np.ndarray],
    cb513_labels: dict,
    cb513_train_ids: list[str],
    cb513_test_ids: list[str],
    external_raw: dict[str, np.ndarray],
    external_labels: dict,
    external_test_ids: list[str],
    external_name: str,
    comp_cb513: dict[str, np.ndarray],
    comp_external: dict[str, np.ndarray],
) -> dict:
    """Train probes on CB513 train set, test on both CB513 test and external.

    For the external test set (TS115 or CASP12), we merge CB513 train
    embeddings with external embeddings into a single dict, then pass
    CB513 train_ids and external protein_ids as test_ids to the runner.

    Args:
        task: "ss3" or "ss8".
        label_map: SS3_MAP or SS8_MAP.
        metric_key: "q3" or "q8" — the key in the runner return dict.
        cb513_raw: Raw CB513 embeddings.
        cb513_labels: CB513 SS labels.
        cb513_train_ids: CB513 training protein IDs.
        cb513_test_ids: CB513 test protein IDs.
        external_raw: External dataset raw embeddings.
        external_labels: External dataset SS labels.
        external_test_ids: External dataset protein IDs (all used as test).
        external_name: Name for reporting (e.g. "TS115").
        comp_cb513: Compressed CB513 per-residue embeddings.
        comp_external: Compressed external per-residue embeddings.

    Returns:
        dict with results for cb513 and external, plus retention.
    """
    runner_fn = run_ss3_benchmark if task == "ss3" else run_ss8_benchmark

    # --- CB513 test set (same as Phase A1) ---
    print(f"\n  --- {task.upper()} on CB513 test ---")
    raw_cb513_result = runner_fn(
        embeddings=cb513_raw,
        labels=cb513_labels,
        train_ids=cb513_train_ids,
        test_ids=cb513_test_ids,
        C_grid=C_GRID,
        cv_folds=CV_FOLDS,
        seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(fmt_metric(f"{metric_key.upper()} raw (CB513)", raw_cb513_result[metric_key]))

    comp_cb513_result = runner_fn(
        embeddings=comp_cb513,
        labels=cb513_labels,
        train_ids=cb513_train_ids,
        test_ids=cb513_test_ids,
        C_grid=C_GRID,
        cv_folds=CV_FOLDS,
        seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(fmt_metric(f"{metric_key.upper()} comp (CB513)", comp_cb513_result[metric_key]))

    cb513_ret_ci = paired_bootstrap_retention(
        raw_cb513_result["per_protein_scores"], comp_cb513_result["per_protein_scores"],
        n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
    )
    cb513_retention = cb513_ret_ci.value
    print(f"  {task.upper()} retention (CB513): {cb513_ret_ci.value:.1f} ± {(cb513_ret_ci.ci_upper - cb513_ret_ci.ci_lower) / 2:.1f}%")

    # --- External test set ---
    # Merge CB513 train embeddings with external embeddings
    # Merge labels too
    print(f"\n  --- {task.upper()} on {external_name} (train on CB513 train) ---")

    merged_raw = {}
    merged_raw.update({pid: cb513_raw[pid] for pid in cb513_train_ids if pid in cb513_raw})
    merged_raw.update(external_raw)

    merged_comp = {}
    merged_comp.update({pid: comp_cb513[pid] for pid in cb513_train_ids if pid in comp_cb513})
    merged_comp.update(comp_external)

    merged_labels = {}
    merged_labels.update(cb513_labels)
    merged_labels.update(external_labels)

    raw_external_result = runner_fn(
        embeddings=merged_raw,
        labels=merged_labels,
        train_ids=cb513_train_ids,
        test_ids=external_test_ids,
        C_grid=C_GRID,
        cv_folds=CV_FOLDS,
        seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(fmt_metric(f"{metric_key.upper()} raw ({external_name})", raw_external_result[metric_key]))

    comp_external_result = runner_fn(
        embeddings=merged_comp,
        labels=merged_labels,
        train_ids=cb513_train_ids,
        test_ids=external_test_ids,
        C_grid=C_GRID,
        cv_folds=CV_FOLDS,
        seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(fmt_metric(f"{metric_key.upper()} comp ({external_name})", comp_external_result[metric_key]))

    ext_ret_ci = paired_bootstrap_retention(
        raw_external_result["per_protein_scores"], comp_external_result["per_protein_scores"],
        n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
    )
    ext_retention = ext_ret_ci.value
    print(f"  {task.upper()} retention ({external_name}): {ext_ret_ci.value:.1f} ± {(ext_ret_ci.ci_upper - ext_ret_ci.ci_lower) / 2:.1f}%")

    return {
        "cb513": {
            "raw": raw_cb513_result,
            "compressed": comp_cb513_result,
            "retention_pct": cb513_retention,
            "retention": cb513_ret_ci,
        },
        external_name.lower(): {
            "raw": raw_external_result,
            "compressed": comp_external_result,
            "retention_pct": ext_retention,
            "retention": ext_ret_ci,
        },
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    all_results = {}

    # Shared: load ABTT corpus (SCOPe 5K) for ProtT5 codec fitting
    scope_corpus_path = RAW_EMBEDDINGS["prot_t5"]
    abtt_corpus_prot_t5 = None
    codec_prot_t5 = None

    if scope_corpus_path.exists():
        print("Loading SCOPe 5K corpus for ABTT fitting (ProtT5)...")
        abtt_corpus_prot_t5 = load_h5_embeddings(scope_corpus_path)
        codec_prot_t5 = Codec(d_out=768, dct_k=4, seed=42)
        codec_prot_t5.fit(abtt_corpus_prot_t5, k=3)
        print(f"  ProtT5 codec fitted on {len(abtt_corpus_prot_t5)} proteins.")
    else:
        print(f"WARNING: SCOPe ProtT5 corpus not found: {scope_corpus_path}")
        print("         Will attempt self-fitting on each dataset (suboptimal).")

    # ==================================================================
    # SECTION 1: SS3/SS8 CROSS-VALIDATION (CB513 + TS115 + CASP12)
    # ==================================================================
    section("1. SS3/SS8 Cross-Validation (ProtT5)")

    cb513_emb_path = RAW_EMBEDDINGS["prot_t5_cb513"]
    cb513_split_path = SPLITS["cb513"]
    cb513_label_path = LABELS["cb513_csv"]

    can_run_ss = (
        cb513_emb_path.exists()
        and cb513_label_path.exists()
        and cb513_split_path.exists()
    )

    if not can_run_ss:
        missing = []
        if not cb513_emb_path.exists():
            missing.append(f"embeddings: {cb513_emb_path}")
        if not cb513_label_path.exists():
            missing.append(f"labels: {cb513_label_path}")
        if not cb513_split_path.exists():
            missing.append(f"split: {cb513_split_path}")
        print(f"  SKIP SS3/SS8: missing files: {', '.join(missing)}")
    else:
        # Load CB513 data
        print("  Loading CB513 data...")
        raw_cb513 = load_h5_embeddings(cb513_emb_path)
        cb513_train_ids, cb513_test_ids = load_split(cb513_split_path)
        sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_label_path)

        # Filter to IDs present in both embeddings and labels
        avail_train_ss3 = [pid for pid in cb513_train_ids if pid in raw_cb513 and pid in ss3_labels]
        avail_test_ss3 = [pid for pid in cb513_test_ids if pid in raw_cb513 and pid in ss3_labels]
        avail_train_ss8 = [pid for pid in cb513_train_ids if pid in raw_cb513 and pid in ss8_labels]
        avail_test_ss8 = [pid for pid in cb513_test_ids if pid in raw_cb513 and pid in ss8_labels]
        print(f"  CB513 — SS3: {len(avail_train_ss3)} train, {len(avail_test_ss3)} test")
        print(f"  CB513 — SS8: {len(avail_train_ss8)} train, {len(avail_test_ss8)} test")

        # Compress CB513
        if codec_prot_t5 is None:
            print("  WARNING: No external corpus; fitting ABTT on CB513 (self-fit).")
            codec_prot_t5 = Codec(d_out=768, dct_k=4, seed=42)
            codec_prot_t5.fit(raw_cb513, k=3)

        print("  Compressing CB513 embeddings...")
        comp_cb513_per_res, _ = compress_embeddings(raw_cb513, codec_prot_t5)

        ss_retention = {}  # {dataset_name: retention_pct} for consistency check

        # ---- TS115 ----
        ts115_available = TS115_EMB.exists() and TS115_CSV.exists()
        if not ts115_available:
            missing_ts115 = []
            if not TS115_EMB.exists():
                missing_ts115.append(f"embeddings: {TS115_EMB}")
            if not TS115_CSV.exists():
                missing_ts115.append(f"labels: {TS115_CSV}")
            print(f"\n  SKIP TS115: missing {', '.join(missing_ts115)}")
            if not TS115_EMB.exists():
                print("  To create TS115 embeddings, run:")
                print(f"    uv run python experiments/01_extract_residue_embeddings.py --dataset ts115")
        else:
            print("\n  Loading TS115 data...")
            raw_ts115 = load_h5_embeddings(TS115_EMB)
            _, ts115_ss3, ts115_ss8 = load_netsurfp_csv(TS115_CSV)

            # Filter to proteins with both embeddings and labels
            ts115_test_ids = [pid for pid in ts115_ss3 if pid in raw_ts115]
            print(f"  TS115: {len(ts115_test_ids)} proteins")

            # Compress TS115
            comp_ts115_per_res, _ = compress_embeddings(raw_ts115, codec_prot_t5)

            # SS3 cross-validation: CB513 + TS115
            ss3_xval = run_cross_dataset_ss_benchmark(
                task="ss3",
                label_map=SS3_MAP,
                metric_key="q3",
                cb513_raw=raw_cb513,
                cb513_labels=ss3_labels,
                cb513_train_ids=avail_train_ss3,
                cb513_test_ids=avail_test_ss3,
                external_raw=raw_ts115,
                external_labels=ts115_ss3,
                external_test_ids=ts115_test_ids,
                external_name="TS115",
                comp_cb513=comp_cb513_per_res,
                comp_external=comp_ts115_per_res,
            )
            all_results["ss3_ts115"] = ss3_xval

            if ss3_xval["cb513"]["retention_pct"] is not None:
                ss_retention["CB513_ss3"] = ss3_xval["cb513"]["retention_pct"]
            if ss3_xval["ts115"]["retention_pct"] is not None:
                ss_retention["TS115_ss3"] = ss3_xval["ts115"]["retention_pct"]

            # SS8 cross-validation: CB513 + TS115
            ss8_xval = run_cross_dataset_ss_benchmark(
                task="ss8",
                label_map=SS8_MAP,
                metric_key="q8",
                cb513_raw=raw_cb513,
                cb513_labels=ss8_labels,
                cb513_train_ids=avail_train_ss8,
                cb513_test_ids=avail_test_ss8,
                external_raw=raw_ts115,
                external_labels=ts115_ss8,
                external_test_ids=ts115_test_ids,
                external_name="TS115",
                comp_cb513=comp_cb513_per_res,
                comp_external=comp_ts115_per_res,
            )
            all_results["ss8_ts115"] = ss8_xval

            if ss8_xval["cb513"]["retention_pct"] is not None:
                ss_retention["CB513_ss8"] = ss8_xval["cb513"]["retention_pct"]
            if ss8_xval["ts115"]["retention_pct"] is not None:
                ss_retention["TS115_ss8"] = ss8_xval["ts115"]["retention_pct"]

        # ---- CASP12 ----
        casp12_available = CASP12_EMB.exists() and CASP12_CSV.exists()
        if not casp12_available:
            missing_casp12 = []
            if not CASP12_EMB.exists():
                missing_casp12.append(f"embeddings: {CASP12_EMB}")
            if not CASP12_CSV.exists():
                missing_casp12.append(f"labels: {CASP12_CSV}")
            print(f"\n  SKIP CASP12: missing {', '.join(missing_casp12)}")
            if not CASP12_EMB.exists():
                print("  To create CASP12 embeddings, run:")
                print(f"    uv run python experiments/01_extract_residue_embeddings.py --dataset casp12")
        else:
            print("\n  Loading CASP12 data...")
            raw_casp12 = load_h5_embeddings(CASP12_EMB)
            _, casp12_ss3, casp12_ss8 = load_netsurfp_csv(CASP12_CSV)

            casp12_test_ids = [pid for pid in casp12_ss3 if pid in raw_casp12]
            print(f"  CASP12: {len(casp12_test_ids)} proteins")

            comp_casp12_per_res, _ = compress_embeddings(raw_casp12, codec_prot_t5)

            # SS3 cross-validation: CB513 + CASP12
            ss3_casp12 = run_cross_dataset_ss_benchmark(
                task="ss3",
                label_map=SS3_MAP,
                metric_key="q3",
                cb513_raw=raw_cb513,
                cb513_labels=ss3_labels,
                cb513_train_ids=avail_train_ss3,
                cb513_test_ids=avail_test_ss3,
                external_raw=raw_casp12,
                external_labels=casp12_ss3,
                external_test_ids=casp12_test_ids,
                external_name="CASP12",
                comp_cb513=comp_cb513_per_res,
                comp_external=comp_casp12_per_res,
            )
            all_results["ss3_casp12"] = ss3_casp12

            if ss3_casp12["casp12"]["retention_pct"] is not None:
                ss_retention["CASP12_ss3"] = ss3_casp12["casp12"]["retention_pct"]

            # SS8 cross-validation: CB513 + CASP12
            ss8_casp12 = run_cross_dataset_ss_benchmark(
                task="ss8",
                label_map=SS8_MAP,
                metric_key="q8",
                cb513_raw=raw_cb513,
                cb513_labels=ss8_labels,
                cb513_train_ids=avail_train_ss8,
                cb513_test_ids=avail_test_ss8,
                external_raw=raw_casp12,
                external_labels=casp12_ss8,
                external_test_ids=casp12_test_ids,
                external_name="CASP12",
                comp_cb513=comp_cb513_per_res,
                comp_external=comp_casp12_per_res,
            )
            all_results["ss8_casp12"] = ss8_casp12

            if ss8_casp12["casp12"]["retention_pct"] is not None:
                ss_retention["CASP12_ss8"] = ss8_casp12["casp12"]["retention_pct"]

        # ---- SS3 cross-dataset consistency ----
        ss3_retentions = {k: v for k, v in ss_retention.items() if "_ss3" in k}
        if len(ss3_retentions) >= 2:
            print("\n  --- SS3 Cross-Dataset Consistency ---")
            ss3_consistency = check_cross_dataset_consistency(
                ss3_retentions,
                warn_pp=CROSS_CHECK_WARN_PP,
                block_pp=CROSS_CHECK_BLOCK_PP,
            )
            print(f"  Max divergence: {ss3_consistency['max_divergence']:.1f} pp")
            print(f"  Status: {ss3_consistency['status']}")
            for a, b, div in ss3_consistency["pairs"]:
                print(f"    {a} vs {b}: {div:.1f} pp")
            all_results["ss3_consistency"] = ss3_consistency

        # ---- SS8 cross-dataset consistency ----
        ss8_retentions = {k: v for k, v in ss_retention.items() if "_ss8" in k}
        if len(ss8_retentions) >= 2:
            print("\n  --- SS8 Cross-Dataset Consistency ---")
            ss8_consistency = check_cross_dataset_consistency(
                ss8_retentions,
                warn_pp=CROSS_CHECK_WARN_PP,
                block_pp=CROSS_CHECK_BLOCK_PP,
            )
            print(f"  Max divergence: {ss8_consistency['max_divergence']:.1f} pp")
            print(f"  Status: {ss8_consistency['status']}")
            for a, b, div in ss8_consistency["pairs"]:
                print(f"    {a} vs {b}: {div:.1f} pp")
            all_results["ss8_consistency"] = ss8_consistency

    # ==================================================================
    # SECTION 2: DISORDER CROSS-VALIDATION (CheZOD + TriZOD)
    # ==================================================================
    section("2. Disorder Cross-Validation (ProtT5)")

    chezod_emb_path = RAW_EMBEDDINGS["prot_t5_chezod"]
    chezod_data_dir = LABELS["chezod_data_dir"]

    can_run_disorder = (
        chezod_emb_path.exists()
        and (chezod_data_dir / "SETH").exists()
    )

    if not can_run_disorder:
        missing = []
        if not chezod_emb_path.exists():
            missing.append(f"embeddings: {chezod_emb_path}")
        if not (chezod_data_dir / "SETH").exists():
            missing.append(f"SETH dir: {chezod_data_dir / 'SETH'}")
        print(f"  SKIP Disorder: missing {', '.join(missing)}")
    else:
        # Load CheZOD data
        print("  Loading CheZOD data...")
        raw_chezod = load_h5_embeddings(chezod_emb_path)
        _, disorder_scores, chezod_train_ids, chezod_test_ids = load_chezod_seth(chezod_data_dir)

        avail_train_dis = [pid for pid in chezod_train_ids if pid in raw_chezod and pid in disorder_scores]
        avail_test_dis = [pid for pid in chezod_test_ids if pid in raw_chezod and pid in disorder_scores]
        print(f"  CheZOD: {len(avail_train_dis)} train, {len(avail_test_dis)} test")

        # Compress CheZOD
        if codec_prot_t5 is None:
            codec_prot_t5 = Codec(d_out=768, dct_k=4, seed=42)
            codec_prot_t5.fit(raw_chezod, k=3)
        comp_chezod_per_res, _ = compress_embeddings(raw_chezod, codec_prot_t5)

        dis_retention = {}

        # ---- CheZOD117 (standard test) ----
        print("\n  --- Disorder on CheZOD117 ---")
        dis_raw_chezod = run_disorder_benchmark(
            embeddings=raw_chezod,
            scores=disorder_scores,
            train_ids=avail_train_dis,
            test_ids=avail_test_dis,
            alpha_grid=ALPHA_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Spearman rho raw (CheZOD117)", dis_raw_chezod["spearman_rho"]))
        print(f"  Pooled rho raw: {dis_raw_chezod['pooled_spearman_rho'].value:.4f}")

        dis_comp_chezod = run_disorder_benchmark(
            embeddings=comp_chezod_per_res,
            scores=disorder_scores,
            train_ids=avail_train_dis,
            test_ids=avail_test_dis,
            alpha_grid=ALPHA_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Spearman rho comp (CheZOD117)", dis_comp_chezod["spearman_rho"]))
        print(f"  Pooled rho comp: {dis_comp_chezod['pooled_spearman_rho'].value:.4f}")

        from scipy.stats import spearmanr as _spearmanr
        def _pooled_spearman(cluster_data):
            all_true = np.concatenate([d["y_true"] for d in cluster_data])
            all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
            rho, _ = _spearmanr(all_true, all_pred)
            return float(rho) if not np.isnan(rho) else 0.0

        chezod_ret_ci = paired_cluster_bootstrap_retention(
            dis_raw_chezod["per_protein_predictions"], dis_comp_chezod["per_protein_predictions"],
            _pooled_spearman, n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        chezod_retention = chezod_ret_ci.value
        print(f"  Disorder retention (CheZOD117): {chezod_ret_ci.value:.1f} ± {(chezod_ret_ci.ci_upper - chezod_ret_ci.ci_lower) / 2:.1f}%")

        all_results["disorder_chezod"] = {
            "raw": dis_raw_chezod,
            "compressed": dis_comp_chezod,
            "retention_pct": chezod_retention,
            "retention": chezod_ret_ci,
        }
        if chezod_retention is not None:
            dis_retention["CheZOD117"] = chezod_retention

        # ---- TriZOD348 (cross-validation) ----
        trizod_can_run = TRIZOD_EMB.exists() and TRIZOD_SPLIT.exists() and TRIZOD_SCORES_JSON.exists()
        if not trizod_can_run:
            missing_tz = []
            if not TRIZOD_EMB.exists():
                missing_tz.append(f"embeddings: {TRIZOD_EMB}")
            if not TRIZOD_SPLIT.exists():
                missing_tz.append(f"split: {TRIZOD_SPLIT}")
            if not TRIZOD_SCORES_JSON.exists():
                missing_tz.append(f"scores: {TRIZOD_SCORES_JSON}")
            print(f"\n  SKIP TriZOD: missing {', '.join(missing_tz)}")
        else:
            print("\n  --- Disorder on TriZOD348 (train CheZOD1174) ---")
            # Load TriZOD embeddings and split
            trizod_data = load_trizod_embeddings(TRIZOD_EMB, TRIZOD_SPLIT)
            raw_trizod = trizod_data["embeddings"]
            trizod_test_ids = trizod_data["test_ids"]

            # Load TriZOD disorder scores from moderate.json
            print("  Loading TriZOD disorder scores from moderate.json...")
            trizod_scores = load_trizod_disorder_scores(TRIZOD_SCORES_JSON)
            print(f"  TriZOD scores loaded for {len(trizod_scores)} proteins")

            # Filter test IDs to those with both embeddings and scores
            trizod_test_avail = [
                pid for pid in trizod_test_ids
                if pid in raw_trizod and pid in trizod_scores
            ]
            print(f"  TriZOD348: {len(trizod_test_avail)} test proteins with scores")

            if len(trizod_test_avail) < 10:
                print(f"  SKIP TriZOD: too few test proteins with scores ({len(trizod_test_avail)})")
            else:
                # Compress TriZOD
                comp_trizod_per_res, _ = compress_embeddings(raw_trizod, codec_prot_t5)

                # Cross-dataset: train on CheZOD1174, test on TriZOD348
                # Merge CheZOD train embeddings + TriZOD test embeddings
                merged_raw_dis = {}
                merged_raw_dis.update({pid: raw_chezod[pid] for pid in avail_train_dis if pid in raw_chezod})
                merged_raw_dis.update({pid: raw_trizod[pid] for pid in trizod_test_avail})

                merged_comp_dis = {}
                merged_comp_dis.update({pid: comp_chezod_per_res[pid] for pid in avail_train_dis if pid in comp_chezod_per_res})
                merged_comp_dis.update({pid: comp_trizod_per_res[pid] for pid in trizod_test_avail})

                # Merge scores: CheZOD + TriZOD
                merged_scores = {}
                merged_scores.update(disorder_scores)
                merged_scores.update(trizod_scores)

                dis_raw_trizod = run_disorder_benchmark(
                    embeddings=merged_raw_dis,
                    scores=merged_scores,
                    train_ids=avail_train_dis,
                    test_ids=trizod_test_avail,
                    alpha_grid=ALPHA_GRID,
                    cv_folds=CV_FOLDS,
                    seeds=SEEDS,
                    n_bootstrap=BOOTSTRAP_N,
                )
                print(fmt_metric("Spearman rho raw (TriZOD348)", dis_raw_trizod["spearman_rho"]))
                print(f"  Pooled rho raw: {dis_raw_trizod['pooled_spearman_rho'].value:.4f}")

                dis_comp_trizod = run_disorder_benchmark(
                    embeddings=merged_comp_dis,
                    scores=merged_scores,
                    train_ids=avail_train_dis,
                    test_ids=trizod_test_avail,
                    alpha_grid=ALPHA_GRID,
                    cv_folds=CV_FOLDS,
                    seeds=SEEDS,
                    n_bootstrap=BOOTSTRAP_N,
                )
                print(fmt_metric("Spearman rho comp (TriZOD348)", dis_comp_trizod["spearman_rho"]))
                print(f"  Pooled rho comp: {dis_comp_trizod['pooled_spearman_rho'].value:.4f}")

                trizod_ret_ci = paired_cluster_bootstrap_retention(
                    dis_raw_trizod["per_protein_predictions"], dis_comp_trizod["per_protein_predictions"],
                    _pooled_spearman, n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
                )
                trizod_retention = trizod_ret_ci.value
                print(f"  Disorder retention (TriZOD348): {trizod_ret_ci.value:.1f} ± {(trizod_ret_ci.ci_upper - trizod_ret_ci.ci_lower) / 2:.1f}%")

                all_results["disorder_trizod"] = {
                    "raw": dis_raw_trizod,
                    "compressed": dis_comp_trizod,
                    "retention_pct": trizod_retention,
                    "retention": trizod_ret_ci,
                }
                if trizod_retention is not None:
                    dis_retention["TriZOD348"] = trizod_retention

        # ---- Disorder cross-dataset consistency ----
        if len(dis_retention) >= 2:
            print("\n  --- Disorder Cross-Dataset Consistency ---")
            dis_consistency = check_cross_dataset_consistency(
                dis_retention,
                warn_pp=CROSS_CHECK_WARN_PP,
                block_pp=CROSS_CHECK_BLOCK_PP,
            )
            print(f"  Max divergence: {dis_consistency['max_divergence']:.1f} pp")
            print(f"  Status: {dis_consistency['status']}")
            for a, b, div in dis_consistency["pairs"]:
                print(f"    {a} vs {b}: {div:.1f} pp")
            all_results["disorder_consistency"] = dis_consistency

    # ==================================================================
    # SECTION 3: ESM2 MULTI-PLM VALIDATION
    # ==================================================================
    section("3. ESM2 Multi-PLM Validation")

    esm2_cb513_exists = ESM2_CB513.exists()
    esm2_scope_exists = ESM2_SCOPE.exists()

    if not esm2_cb513_exists and not esm2_scope_exists:
        print(f"  SKIP ESM2: no ESM2 embeddings found.")
        if not esm2_cb513_exists:
            print(f"    Missing: {ESM2_CB513}")
        if not esm2_scope_exists:
            print(f"    Missing: {ESM2_SCOPE}")
    else:
        # Fit ESM2 codec on ESM2 SCOPe 5K corpus
        esm2_codec = None
        esm2_corpus = None

        if esm2_scope_exists:
            print("  Loading ESM2 SCOPe 5K corpus for ABTT fitting...")
            esm2_corpus = load_h5_embeddings(ESM2_SCOPE)
            esm2_codec = Codec(d_out=768, dct_k=4, seed=42)
            esm2_codec.fit(esm2_corpus, k=3)
            print(f"  ESM2 codec fitted on {len(esm2_corpus)} proteins.")

        # ---- ESM2 SS3/SS8 on CB513 ----
        if esm2_cb513_exists and can_run_ss:
            print("\n  --- ESM2 SS3/SS8 on CB513 ---")
            raw_esm2_cb513 = load_h5_embeddings(ESM2_CB513)

            # Check dimensionality
            sample_key = next(iter(raw_esm2_cb513))
            esm2_dim = raw_esm2_cb513[sample_key].shape[1]
            print(f"  ESM2 embedding dim: {esm2_dim}")

            # Re-use CB513 labels and split (same proteins, different PLM)
            # Filter IDs to those present in ESM2 embeddings
            esm2_train_ss3 = [pid for pid in cb513_train_ids if pid in raw_esm2_cb513 and pid in ss3_labels]
            esm2_test_ss3 = [pid for pid in cb513_test_ids if pid in raw_esm2_cb513 and pid in ss3_labels]
            esm2_train_ss8 = [pid for pid in cb513_train_ids if pid in raw_esm2_cb513 and pid in ss8_labels]
            esm2_test_ss8 = [pid for pid in cb513_test_ids if pid in raw_esm2_cb513 and pid in ss8_labels]
            print(f"  ESM2 CB513 — SS3: {len(esm2_train_ss3)} train, {len(esm2_test_ss3)} test")
            print(f"  ESM2 CB513 — SS8: {len(esm2_train_ss8)} train, {len(esm2_test_ss8)} test")

            if len(esm2_test_ss3) < 10:
                print(f"  SKIP ESM2 SS3: too few proteins ({len(esm2_test_ss3)})")
            else:
                # Fit codec on ESM2 CB513 if no corpus available
                if esm2_codec is None:
                    print("  WARNING: No ESM2 corpus; fitting ABTT on ESM2 CB513 (self-fit).")
                    esm2_codec = Codec(d_out=768, dct_k=4, seed=42)
                    esm2_codec.fit(raw_esm2_cb513, k=3)

                # Compress ESM2 CB513
                print("  Compressing ESM2 CB513 embeddings...")
                comp_esm2_cb513, _ = compress_embeddings(raw_esm2_cb513, esm2_codec)

                # SS3 on ESM2
                print("  Running SS3 on ESM2 raw...")
                ss3_esm2_raw = run_ss3_benchmark(
                    embeddings=raw_esm2_cb513,
                    labels=ss3_labels,
                    train_ids=esm2_train_ss3,
                    test_ids=esm2_test_ss3,
                    C_grid=C_GRID,
                    cv_folds=CV_FOLDS,
                    seeds=SEEDS,
                    n_bootstrap=BOOTSTRAP_N,
                )
                print(fmt_metric(f"Q3 raw (ESM2 {esm2_dim}d)", ss3_esm2_raw["q3"]))

                print("  Running SS3 on ESM2 compressed...")
                ss3_esm2_comp = run_ss3_benchmark(
                    embeddings=comp_esm2_cb513,
                    labels=ss3_labels,
                    train_ids=esm2_train_ss3,
                    test_ids=esm2_test_ss3,
                    C_grid=C_GRID,
                    cv_folds=CV_FOLDS,
                    seeds=SEEDS,
                    n_bootstrap=BOOTSTRAP_N,
                )
                print(fmt_metric("Q3 comp (ESM2 768d)", ss3_esm2_comp["q3"]))

                esm2_ss3_ret_ci = paired_bootstrap_retention(
                    ss3_esm2_raw["per_protein_scores"], ss3_esm2_comp["per_protein_scores"],
                    n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
                )
                esm2_ss3_retention = esm2_ss3_ret_ci.value
                print(f"  ESM2 SS3 retention: {esm2_ss3_ret_ci.value:.1f} ± {(esm2_ss3_ret_ci.ci_upper - esm2_ss3_ret_ci.ci_lower) / 2:.1f}%")

                all_results["esm2_ss3"] = {
                    "raw": ss3_esm2_raw,
                    "compressed": ss3_esm2_comp,
                    "retention_pct": esm2_ss3_retention,
                    "retention": esm2_ss3_ret_ci,
                    "esm2_dim": esm2_dim,
                }

                # SS8 on ESM2
                if len(esm2_test_ss8) >= 10:
                    print("\n  Running SS8 on ESM2 raw...")
                    ss8_esm2_raw = run_ss8_benchmark(
                        embeddings=raw_esm2_cb513,
                        labels=ss8_labels,
                        train_ids=esm2_train_ss8,
                        test_ids=esm2_test_ss8,
                        C_grid=C_GRID,
                        cv_folds=CV_FOLDS,
                        seeds=SEEDS,
                        n_bootstrap=BOOTSTRAP_N,
                    )
                    print(fmt_metric(f"Q8 raw (ESM2 {esm2_dim}d)", ss8_esm2_raw["q8"]))

                    print("  Running SS8 on ESM2 compressed...")
                    ss8_esm2_comp = run_ss8_benchmark(
                        embeddings=comp_esm2_cb513,
                        labels=ss8_labels,
                        train_ids=esm2_train_ss8,
                        test_ids=esm2_test_ss8,
                        C_grid=C_GRID,
                        cv_folds=CV_FOLDS,
                        seeds=SEEDS,
                        n_bootstrap=BOOTSTRAP_N,
                    )
                    print(fmt_metric("Q8 comp (ESM2 768d)", ss8_esm2_comp["q8"]))

                    esm2_ss8_ret_ci = paired_bootstrap_retention(
                        ss8_esm2_raw["per_protein_scores"], ss8_esm2_comp["per_protein_scores"],
                        n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
                    )
                    esm2_ss8_retention = esm2_ss8_ret_ci.value
                    print(f"  ESM2 SS8 retention: {esm2_ss8_ret_ci.value:.1f} ± {(esm2_ss8_ret_ci.ci_upper - esm2_ss8_ret_ci.ci_lower) / 2:.1f}%")

                    all_results["esm2_ss8"] = {
                        "raw": ss8_esm2_raw,
                        "compressed": ss8_esm2_comp,
                        "retention_pct": esm2_ss8_retention,
                        "retention": esm2_ss8_ret_ci,
                        "esm2_dim": esm2_dim,
                    }
        elif esm2_cb513_exists and not can_run_ss:
            print("  SKIP ESM2 SS3/SS8: CB513 labels/split not available.")

        # ---- ESM2 Retrieval on SCOPe 5K ----
        scope_meta_path = METADATA["scope_5k"]
        if esm2_scope_exists and scope_meta_path.exists():
            print("\n  --- ESM2 Retrieval on SCOPe 5K ---")
            if esm2_corpus is None:
                esm2_corpus = load_h5_embeddings(ESM2_SCOPE)
            metadata = load_scope_metadata(scope_meta_path)

            # Check dimensionality
            sample_key = next(iter(esm2_corpus))
            esm2_dim = esm2_corpus[sample_key].shape[1]
            print(f"  ESM2 embedding dim: {esm2_dim}")

            # Raw ESM2 + DCT K=4 (fair baseline)
            print("  Computing ESM2 raw protein vectors (DCT K=4)...")
            stats_esm2 = compute_corpus_stats(esm2_corpus, n_pcs=3, seed=42)
            abtt_esm2 = apply_abtt_to_dict(esm2_corpus, stats_esm2)
            vecs_esm2_raw = compute_protein_vectors(abtt_esm2, method="dct_k4", dct_k=4)

            ret_esm2_raw = run_retrieval_benchmark(
                vectors=vecs_esm2_raw,
                metadata=metadata,
                label_key="family",
                n_bootstrap=BOOTSTRAP_N,
                seed=42,
            )
            print(fmt_metric("Ret@1 cosine  (ESM2 ABTT3+DCT4)", ret_esm2_raw["ret1_cosine"]))
            print(fmt_metric("Ret@1 euclid  (ESM2 ABTT3+DCT4)", ret_esm2_raw["ret1_euclidean"]))

            # Compressed ESM2 protein_vec
            if esm2_codec is None:
                esm2_codec = Codec(d_out=768, dct_k=4, seed=42)
                esm2_codec.fit(esm2_corpus, k=3)

            print("  Compressing ESM2 SCOPe embeddings...")
            _, comp_esm2_vecs = compress_embeddings(esm2_corpus, esm2_codec)

            ret_esm2_comp = run_retrieval_benchmark(
                vectors=comp_esm2_vecs,
                metadata=metadata,
                label_key="family",
                n_bootstrap=BOOTSTRAP_N,
                seed=42,
            )
            print(fmt_metric("Ret@1 cosine  (ESM2 compressed)", ret_esm2_comp["ret1_cosine"]))
            print(fmt_metric("Ret@1 euclid  (ESM2 compressed)", ret_esm2_comp["ret1_euclidean"]))

            esm2_ret_cos_ci = paired_bootstrap_retention(
                ret_esm2_raw["per_query_cosine"], ret_esm2_comp["per_query_cosine"],
                n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
            )
            esm2_ret_euc_ci = paired_bootstrap_retention(
                ret_esm2_raw["per_query_euclidean"], ret_esm2_comp["per_query_euclidean"],
                n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
            )
            esm2_ret_cosine_retention = esm2_ret_cos_ci.value
            esm2_ret_euclidean_retention = esm2_ret_euc_ci.value
            print(f"  ESM2 Ret@1 cosine retention: {esm2_ret_cos_ci.value:.1f} ± {(esm2_ret_cos_ci.ci_upper - esm2_ret_cos_ci.ci_lower) / 2:.1f}%")

            print(fmt_retention("ESM2 Ret@1 cosine retention",
                                ret_esm2_comp["ret1_cosine"].value,
                                ret_esm2_raw["ret1_cosine"].value))
            print(fmt_retention("ESM2 Ret@1 euclidean retention",
                                ret_esm2_comp["ret1_euclidean"].value,
                                ret_esm2_raw["ret1_euclidean"].value))

            all_results["esm2_retrieval"] = {
                "baseline_abtt3_dct4": ret_esm2_raw,
                "compressed": ret_esm2_comp,
                "retention_cosine_pct": esm2_ret_cosine_retention,
                "retention_euclidean_pct": esm2_ret_euclidean_retention,
                "esm2_dim": esm2_dim,
            }
        elif not esm2_scope_exists:
            print(f"  SKIP ESM2 retrieval: missing {ESM2_SCOPE}")
        elif not scope_meta_path.exists():
            print(f"  SKIP ESM2 retrieval: missing {scope_meta_path}")

    # ==================================================================
    # SECTION 4: SUMMARY
    # ==================================================================
    section("4. Phase B Summary")

    summary_rows = []

    # SS3/SS8 cross-validation results
    for task_key, metric_key in [("ss3", "SS3 Q3"), ("ss8", "SS8 Q8")]:
        for dataset_key, dataset_name in [
            (f"{task_key}_ts115", "TS115"),
            (f"{task_key}_casp12", "CASP12"),
        ]:
            if dataset_key in all_results:
                # CB513 result (internal validation)
                cb_ret = all_results[dataset_key]["cb513"].get("retention_pct")
                if cb_ret is not None:
                    summary_rows.append((f"{metric_key} (CB513)", cb_ret, "ProtT5, internal"))

                # External result
                ext_key = dataset_name.lower()
                ext_ret = all_results[dataset_key].get(ext_key, {}).get("retention_pct")
                if ext_ret is not None:
                    summary_rows.append((f"{metric_key} ({dataset_name})", ext_ret, "ProtT5, cross-val"))

    # Disorder results
    if "disorder_chezod" in all_results:
        ret = all_results["disorder_chezod"].get("retention_pct")
        if ret is not None:
            summary_rows.append(("Disorder rho (CheZOD117)", ret, "ProtT5, internal"))
    if "disorder_trizod" in all_results:
        ret = all_results["disorder_trizod"].get("retention_pct")
        if ret is not None:
            summary_rows.append(("Disorder rho (TriZOD348)", ret, "ProtT5, cross-val"))

    # ESM2 results
    if "esm2_ss3" in all_results:
        ret = all_results["esm2_ss3"].get("retention_pct")
        if ret is not None:
            summary_rows.append(("SS3 Q3 (ESM2 CB513)", ret, "ESM2, internal"))
    if "esm2_ss8" in all_results:
        ret = all_results["esm2_ss8"].get("retention_pct")
        if ret is not None:
            summary_rows.append(("SS8 Q8 (ESM2 CB513)", ret, "ESM2, internal"))
    if "esm2_retrieval" in all_results:
        ret_cos = all_results["esm2_retrieval"].get("retention_cosine_pct")
        ret_euc = all_results["esm2_retrieval"].get("retention_euclidean_pct")
        if ret_cos is not None:
            summary_rows.append(("Ret@1 cosine (ESM2 SCOPe)", ret_cos, "ESM2"))
        if ret_euc is not None:
            summary_rows.append(("Ret@1 euclidean (ESM2 SCOPe)", ret_euc, "ESM2"))

    if summary_rows:
        # Deduplicate: for the summary table, take only unique (task, dataset) pairs
        seen = set()
        unique_rows = []
        for r in summary_rows:
            if r[0] not in seen:
                seen.add(r[0])
                unique_rows.append(r)
        summary_rows = unique_rows

        print(f"\n  {'Task':<35} {'Retention':>10}   {'Notes'}")
        print(f"  {'-'*35} {'-'*10}   {'-'*30}")
        for task, ret, notes in summary_rows:
            print(f"  {task:<35} {ret:>9.1f}%   {notes}")

        valid = [r[1] for r in summary_rows if r[1] is not None]
        if valid:
            mean_ret = np.mean(valid)
            std_ret = np.std(valid, ddof=1) if len(valid) > 1 else 0.0
            print(f"\n  Mean retention (all): {mean_ret:.1f}% +/- {std_ret:.1f}%")

            # Separate ProtT5 and ESM2 means
            prot_t5_rets = [r[1] for r in summary_rows if r[1] is not None and "ESM2" not in r[2]]
            esm2_rets = [r[1] for r in summary_rows if r[1] is not None and "ESM2" in r[2]]

            if prot_t5_rets:
                print(f"  Mean retention (ProtT5): {np.mean(prot_t5_rets):.1f}%")
            if esm2_rets:
                print(f"  Mean retention (ESM2):   {np.mean(esm2_rets):.1f}%")

            all_results["summary"] = {
                "tasks": [
                    {"task": r[0], "retention_pct": r[1], "notes": r[2]}
                    for r in summary_rows
                ],
                "mean_retention_pct": float(mean_ret),
                "std_retention_pct": float(std_ret),
                "prot_t5_mean": float(np.mean(prot_t5_rets)) if prot_t5_rets else None,
                "esm2_mean": float(np.mean(esm2_rets)) if esm2_rets else None,
            }
    else:
        print("  No benchmarks completed.")

    # ---- Cross-dataset consistency summary ----
    consistency_checks = []
    for key in ["ss3_consistency", "ss8_consistency", "disorder_consistency"]:
        if key in all_results:
            consistency_checks.append((key, all_results[key]))

    if consistency_checks:
        print("\n  --- Cross-Dataset Consistency Verdicts ---")
        for name, check in consistency_checks:
            status_emoji = {"ok": "PASS", "warn": "WARN", "block": "BLOCK"}
            print(f"  {name}: {status_emoji.get(check['status'], check['status'])} "
                  f"(max divergence: {check['max_divergence']:.1f} pp)")

    # ---- Multi-PLM agreement ----
    if "esm2_ss3" in all_results and "ss3_ts115" in all_results:
        prot_t5_ss3_ret = all_results["ss3_ts115"]["cb513"].get("retention_pct")
        esm2_ss3_ret = all_results["esm2_ss3"].get("retention_pct")
        if prot_t5_ss3_ret is not None and esm2_ss3_ret is not None:
            plm_delta = abs(prot_t5_ss3_ret - esm2_ss3_ret)
            print(f"\n  Multi-PLM SS3 agreement: delta = {plm_delta:.1f} pp "
                  f"(ProtT5: {prot_t5_ss3_ret:.1f}%, ESM2: {esm2_ss3_ret:.1f}%)")
            all_results["multi_plm_agreement"] = {
                "ss3_delta_pp": plm_delta,
                "prot_t5_ss3_retention": prot_t5_ss3_ret,
                "esm2_ss3_retention": esm2_ss3_ret,
            }

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
    section("5. Save Results")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "phase_b_results.json"

    serializable = results_to_serializable(all_results)
    serializable["_meta"] = {
        "script": "run_phase_b.py",
        "phase": "B — Cross-validation + Multi-PLM",
        "seeds": SEEDS,
        "bootstrap_n": BOOTSTRAP_N,
        "cv_folds": CV_FOLDS,
        "C_grid": C_GRID,
        "alpha_grid": ALPHA_GRID,
        "codec": "ABTT3 + RP768 + DCT K=4 (seed=42)",
        "cross_check_warn_pp": CROSS_CHECK_WARN_PP,
        "cross_check_block_pp": CROSS_CHECK_BLOCK_PP,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
