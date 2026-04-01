#!/usr/bin/env python
"""Phase A1: Corrected retention benchmarks with fair baselines.

Benchmarks the One Embedding 1.0 codec (ABTT3 + RP768 + DCT K=4) against
raw ProtT5 embeddings on four tasks:

    1. SS3  (Q3)        — CB513, LogReg probe, multi-seed, bootstrap CI
    2. SS8  (Q8)        — CB513, LogReg probe, multi-seed, bootstrap CI
    3. Disorder (rho)   — CheZOD, Ridge probe, multi-seed, bootstrap CI
    4. Family Ret@1     — SCOPe 5K, cosine + euclidean, bootstrap CI

Retrieval uses THREE fair baselines to isolate each pipeline stage:
    A: Raw + mean pool          (context — what most papers report)
    B: Raw + DCT K=4            (fair pooling comparison)
    C: Raw + ABTT3 + DCT K=4   (full pipeline minus RP)
    Compressed: Codec output protein_vec

Retention is computed as Compressed / Baseline C (same pooling, same
preprocessing — only RP is different).

Results saved to RESULTS_DIR / "phase_a1_results.json".

Usage:
    uv run python experiments/43_rigorous_benchmark/run_phase_a1.py
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
)
from rules import MetricResult
from metrics.statistics import paired_bootstrap_retention, paired_cluster_bootstrap_retention

# Runners (from experiment dir)
from runners.per_residue import (
    run_disorder_benchmark,
    run_ss3_benchmark,
    run_ss8_benchmark,
)
from runners.protein_level import (
    compute_protein_vectors,
    run_retrieval_benchmark,
)

# Project-level imports
from src.one_embedding.core.codec import Codec
from src.one_embedding.preprocessing import (
    all_but_the_top,
    compute_corpus_stats,
)
from src.evaluation.per_residue_tasks import load_cb513_csv, load_chezod_seth


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_metric(name: str, mr: MetricResult, indent: int = 2) -> str:
    """Format a MetricResult for console display.

    Example output:
        Q3 (raw_1024d): 0.8465 (95% CI: [0.8312, 0.8618], n=103) [seeds: 0.8460 +/- 0.0012]
    """
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


def load_split(path: Path) -> tuple[list[str], list[str]]:
    """Load train/test split from JSON. Keys: 'train_ids', 'test_ids'."""
    with open(path) as f:
        data = json.load(f)
    return data["train_ids"], data["test_ids"]


def load_scope_metadata(path: Path) -> list[dict]:
    """Load SCOPe metadata CSV as list of dicts."""
    df = pd.read_csv(path)
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------

def compress_embeddings(
    raw_embeddings: dict[str, np.ndarray],
    codec: Codec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compress all embeddings with the fitted codec.

    Returns:
        (per_residue_dict, protein_vec_dict) where:
            per_residue_dict: {pid: (L, d_out) float32}
            protein_vec_dict: {pid: (dct_k * d_out,) float32}
    """
    per_residue = {}
    protein_vecs = {}
    for pid, raw in raw_embeddings.items():
        encoded = codec.encode(raw)
        # Upcast from fp16 to fp32 for probes
        per_residue[pid] = encoded["per_residue"].astype(np.float32)
        protein_vecs[pid] = encoded["protein_vec"].astype(np.float32)
    return per_residue, protein_vecs


def apply_abtt_to_dict(
    embeddings: dict[str, np.ndarray],
    stats: dict,
) -> dict[str, np.ndarray]:
    """Apply ABTT3 preprocessing (center + remove top PCs) to each protein.

    Args:
        embeddings: {pid: (L, D)} raw embeddings.
        stats: Output of compute_corpus_stats (needs 'mean_vec', 'top_pcs').

    Returns:
        {pid: (L, D)} preprocessed embeddings.
    """
    mean_vec = stats["mean_vec"]
    top_pcs = stats["top_pcs"]
    result = {}
    for pid, emb in embeddings.items():
        centered = emb - mean_vec
        result[pid] = all_but_the_top(centered, top_pcs)
    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def metric_to_dict(mr: MetricResult) -> dict:
    """Convert MetricResult to JSON-serializable dict."""
    return asdict(mr)


# Keys with large per-protein data (not needed in JSON output)
_SKIP_KEYS = {"per_protein_scores", "per_protein_predictions", "per_query_cosine", "per_query_euclidean"}


def results_to_serializable(results: dict) -> dict:
    """Recursively convert a results dict so MetricResults become dicts."""
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
        else:
            out[k] = v
    return out


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    all_results = {}

    # ==================================================================
    # 1. SS3 BENCHMARK (CB513)
    # ==================================================================
    section("1. SS3 Benchmark (CB513)")

    cb513_emb_path = RAW_EMBEDDINGS["prot_t5_cb513"]
    cb513_split_path = SPLITS["cb513"]
    cb513_label_path = LABELS["cb513_csv"]

    if not cb513_emb_path.exists():
        print(f"  SKIP: raw embeddings not found: {cb513_emb_path}")
    elif not cb513_label_path.exists():
        print(f"  SKIP: labels not found: {cb513_label_path}")
    elif not cb513_split_path.exists():
        print(f"  SKIP: split not found: {cb513_split_path}")
    else:
        # Load data
        print("  Loading CB513 embeddings...")
        raw_cb513 = load_h5_embeddings(cb513_emb_path)
        train_ids, test_ids = load_split(cb513_split_path)
        sequences, ss3_labels, ss8_labels, disorder_labels = load_cb513_csv(cb513_label_path)

        # Filter to IDs present in both embeddings and labels
        available_train = [pid for pid in train_ids if pid in raw_cb513 and pid in ss3_labels]
        available_test = [pid for pid in test_ids if pid in raw_cb513 and pid in ss3_labels]
        print(f"  Train: {len(available_train)}, Test: {len(available_test)}")

        # --- Raw 1024d ---
        print("  Running SS3 on raw 1024d...")
        ss3_raw = run_ss3_benchmark(
            embeddings=raw_cb513,
            labels=ss3_labels,
            train_ids=available_train,
            test_ids=available_test,
            C_grid=C_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Q3 (raw_1024d)", ss3_raw["q3"]))
        print(f"  Class balance: {ss3_raw['class_balance']}")
        print(f"  Best C: {ss3_raw['best_C']}")

        # --- Compressed 768d ---
        # Fit ABTT on SCOPe 5K corpus (general-purpose, not CB513 itself)
        scope_corpus_path = RAW_EMBEDDINGS["prot_t5"]
        if scope_corpus_path.exists():
            print("  Fitting codec on SCOPe 5K corpus (external)...")
            abtt_corpus = load_h5_embeddings(scope_corpus_path)
        else:
            print("  WARNING: SCOPe corpus not found, fitting ABTT on CB513 (self-fit)...")
            abtt_corpus = raw_cb513
        codec = Codec(d_out=768, dct_k=4, seed=42)
        codec.fit(abtt_corpus, k=3)

        comp_cb513_per_res, _ = compress_embeddings(raw_cb513, codec)

        print("  Running SS3 on compressed 768d...")
        ss3_comp = run_ss3_benchmark(
            embeddings=comp_cb513_per_res,
            labels=ss3_labels,
            train_ids=available_train,
            test_ids=available_test,
            C_grid=C_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Q3 (compressed_768d)", ss3_comp["q3"]))
        print(fmt_retention("SS3 retention", ss3_comp["q3"].value, ss3_raw["q3"].value))

        ss3_ret_ci = paired_bootstrap_retention(
            ss3_raw["per_protein_scores"], ss3_comp["per_protein_scores"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"  SS3 retention: {ss3_ret_ci.value:.1f} ± {(ss3_ret_ci.ci_upper - ss3_ret_ci.ci_lower) / 2:.1f}%")
        all_results["ss3"] = {
            "raw": ss3_raw,
            "compressed": ss3_comp,
            "retention_pct": ss3_ret_ci.value,
            "retention": ss3_ret_ci,
        }

        # ==============================================================
        # 2. SS8 BENCHMARK (CB513, same data)
        # ==============================================================
        section("2. SS8 Benchmark (CB513)")

        available_train_ss8 = [pid for pid in train_ids if pid in raw_cb513 and pid in ss8_labels]
        available_test_ss8 = [pid for pid in test_ids if pid in raw_cb513 and pid in ss8_labels]
        print(f"  Train: {len(available_train_ss8)}, Test: {len(available_test_ss8)}")

        # --- Raw 1024d ---
        print("  Running SS8 on raw 1024d...")
        ss8_raw = run_ss8_benchmark(
            embeddings=raw_cb513,
            labels=ss8_labels,
            train_ids=available_train_ss8,
            test_ids=available_test_ss8,
            C_grid=C_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Q8 (raw_1024d)", ss8_raw["q8"]))
        print(f"  Class balance: {ss8_raw['class_balance']}")
        print(f"  Best C: {ss8_raw['best_C']}")

        # --- Compressed 768d ---
        print("  Running SS8 on compressed 768d...")
        ss8_comp = run_ss8_benchmark(
            embeddings=comp_cb513_per_res,
            labels=ss8_labels,
            train_ids=available_train_ss8,
            test_ids=available_test_ss8,
            C_grid=C_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Q8 (compressed_768d)", ss8_comp["q8"]))
        print(fmt_retention("SS8 retention", ss8_comp["q8"].value, ss8_raw["q8"].value))

        ss8_ret_ci = paired_bootstrap_retention(
            ss8_raw["per_protein_scores"], ss8_comp["per_protein_scores"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"  SS8 retention: {ss8_ret_ci.value:.1f} ± {(ss8_ret_ci.ci_upper - ss8_ret_ci.ci_lower) / 2:.1f}%")
        all_results["ss8"] = {
            "raw": ss8_raw,
            "compressed": ss8_comp,
            "retention_pct": ss8_ret_ci.value,
            "retention": ss8_ret_ci,
        }

    # ==================================================================
    # 3. DISORDER BENCHMARK (CheZOD)
    # ==================================================================
    section("3. Disorder Benchmark (CheZOD)")

    chezod_emb_path = RAW_EMBEDDINGS["prot_t5_chezod"]
    chezod_data_dir = LABELS["chezod_data_dir"]

    if not chezod_emb_path.exists():
        print(f"  SKIP: raw embeddings not found: {chezod_emb_path}")
    elif not (chezod_data_dir / "SETH").exists():
        print(f"  SKIP: SETH directory not found: {chezod_data_dir / 'SETH'}")
    else:
        # Load data
        print("  Loading CheZOD embeddings...")
        raw_chezod = load_h5_embeddings(chezod_emb_path)

        print("  Loading CheZOD labels via load_chezod_seth...")
        sequences_cz, disorder_scores, chezod_train_ids, chezod_test_ids = load_chezod_seth(
            chezod_data_dir
        )

        # Filter to IDs present in both embeddings and labels
        available_train_dis = [pid for pid in chezod_train_ids if pid in raw_chezod and pid in disorder_scores]
        available_test_dis = [pid for pid in chezod_test_ids if pid in raw_chezod and pid in disorder_scores]
        print(f"  Train: {len(available_train_dis)}, Test: {len(available_test_dis)}")

        # --- Raw 1024d ---
        print("  Running disorder on raw 1024d...")
        dis_raw = run_disorder_benchmark(
            embeddings=raw_chezod,
            scores=disorder_scores,
            train_ids=available_train_dis,
            test_ids=available_test_dis,
            alpha_grid=ALPHA_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Spearman rho (raw_1024d) [per-protein avg]", dis_raw["spearman_rho"]))
        print(f"  Pooled residue-level rho (raw_1024d): {dis_raw['pooled_spearman_rho'].value:.4f}")
        print(f"  Best alpha: {dis_raw['best_alpha']}")

        # --- Compressed 768d ---
        # Fit ABTT on SCOPe 5K corpus (general-purpose, not CheZOD itself)
        scope_corpus_path = RAW_EMBEDDINGS["prot_t5"]
        if scope_corpus_path.exists():
            print("  Fitting codec on SCOPe 5K corpus (external)...")
            if "abtt_corpus" not in dir():
                abtt_corpus = load_h5_embeddings(scope_corpus_path)
        else:
            print("  WARNING: SCOPe corpus not found, fitting ABTT on CheZOD (self-fit)...")
            abtt_corpus = raw_chezod
        codec_cz = Codec(d_out=768, dct_k=4, seed=42)
        codec_cz.fit(abtt_corpus, k=3)

        comp_chezod_per_res, _ = compress_embeddings(raw_chezod, codec_cz)

        print("  Running disorder on compressed 768d...")
        dis_comp = run_disorder_benchmark(
            embeddings=comp_chezod_per_res,
            scores=disorder_scores,
            train_ids=available_train_dis,
            test_ids=available_test_dis,
            alpha_grid=ALPHA_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(fmt_metric("Spearman rho (compressed_768d) [per-protein avg]", dis_comp["spearman_rho"]))
        print(f"  Pooled residue-level rho (compressed_768d): {dis_comp['pooled_spearman_rho'].value:.4f}")
        print(fmt_retention("Disorder retention (per-protein)", dis_comp["spearman_rho"].value, dis_raw["spearman_rho"].value))
        print(fmt_retention("Disorder retention (pooled)", dis_comp["pooled_spearman_rho"].value, dis_raw["pooled_spearman_rho"].value))

        # Paired cluster bootstrap retention for disorder (pooled rho)
        from runners.per_residue import pooled_spearman

        dis_ret_ci = paired_cluster_bootstrap_retention(
            dis_raw["per_protein_predictions"], dis_comp["per_protein_predictions"],
            pooled_spearman, n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"  Disorder pooled retention: {dis_ret_ci.value:.1f} ± {(dis_ret_ci.ci_upper - dis_ret_ci.ci_lower) / 2:.1f}%")

        all_results["disorder"] = {
            "raw": dis_raw,
            "compressed": dis_comp,
            "retention_pct": dis_ret_ci.value,
            "retention": dis_ret_ci,
        }

    # ==================================================================
    # 4. FAMILY RETRIEVAL (SCOPe 5K) — 3 FAIR BASELINES
    # ==================================================================
    section("4. Family Retrieval (SCOPe 5K)")

    scope_emb_path = RAW_EMBEDDINGS["prot_t5"]
    scope_meta_path = METADATA["scope_5k"]

    if not scope_emb_path.exists():
        print(f"  SKIP: raw embeddings not found: {scope_emb_path}")
    elif not scope_meta_path.exists():
        print(f"  SKIP: metadata not found: {scope_meta_path}")
    else:
        # Load data
        print("  Loading SCOPe 5K embeddings...")
        raw_scope = load_h5_embeddings(scope_emb_path)
        metadata = load_scope_metadata(scope_meta_path)

        # Filter metadata to only proteins with embeddings
        meta_ids = {m["id"] for m in metadata}
        available_scope = [pid for pid in raw_scope if pid in meta_ids]
        print(f"  Proteins with embeddings + metadata: {len(available_scope)}")

        # ---- Baseline A: Raw + mean pool ----
        print("\n  --- Baseline A: Raw + mean pool ---")
        vecs_A = compute_protein_vectors(raw_scope, method="mean")
        ret_A = run_retrieval_benchmark(
            vectors=vecs_A,
            metadata=metadata,
            label_key="family",
            n_bootstrap=BOOTSTRAP_N,
            seed=42,
        )
        print(fmt_metric("Ret@1 cosine  (raw+mean)", ret_A["ret1_cosine"]))
        print(fmt_metric("Ret@1 euclid  (raw+mean)", ret_A["ret1_euclidean"]))

        # ---- Baseline B: Raw + DCT K=4 ----
        print("\n  --- Baseline B: Raw + DCT K=4 ---")
        vecs_B = compute_protein_vectors(raw_scope, method="dct_k4", dct_k=4)
        ret_B = run_retrieval_benchmark(
            vectors=vecs_B,
            metadata=metadata,
            label_key="family",
            n_bootstrap=BOOTSTRAP_N,
            seed=42,
        )
        print(fmt_metric("Ret@1 cosine  (raw+dct4)", ret_B["ret1_cosine"]))
        print(fmt_metric("Ret@1 euclid  (raw+dct4)", ret_B["ret1_euclidean"]))

        # ---- Baseline C: Raw + ABTT3 + DCT K=4 ----
        print("\n  --- Baseline C: Raw + ABTT3 + DCT K=4 ---")
        print("  Computing corpus stats for ABTT3...")
        stats = compute_corpus_stats(raw_scope, n_pcs=3, seed=42)
        abtt_scope = apply_abtt_to_dict(raw_scope, stats)

        vecs_C = compute_protein_vectors(abtt_scope, method="dct_k4", dct_k=4)
        ret_C = run_retrieval_benchmark(
            vectors=vecs_C,
            metadata=metadata,
            label_key="family",
            n_bootstrap=BOOTSTRAP_N,
            seed=42,
        )
        print(fmt_metric("Ret@1 cosine  (abtt3+dct4)", ret_C["ret1_cosine"]))
        print(fmt_metric("Ret@1 euclid  (abtt3+dct4)", ret_C["ret1_euclidean"]))

        # ---- Compressed: Codec output protein_vec ----
        print("\n  --- Compressed: Codec (ABTT3 + RP768 + DCT K=4) ---")
        print("  Fitting codec on SCOPe corpus...")
        codec_scope = Codec(d_out=768, dct_k=4, seed=42)
        codec_scope.fit(raw_scope, k=3)

        _, comp_scope_vecs = compress_embeddings(raw_scope, codec_scope)
        ret_comp = run_retrieval_benchmark(
            vectors=comp_scope_vecs,
            metadata=metadata,
            label_key="family",
            n_bootstrap=BOOTSTRAP_N,
            seed=42,
        )
        print(fmt_metric("Ret@1 cosine  (compressed)", ret_comp["ret1_cosine"]))
        print(fmt_metric("Ret@1 euclid  (compressed)", ret_comp["ret1_euclidean"]))

        # ---- Retention vs Baseline C ----
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

        ret_cos_ci = paired_bootstrap_retention(
            ret_C["per_query_cosine"], ret_comp["per_query_cosine"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        ret_euc_ci = paired_bootstrap_retention(
            ret_C["per_query_euclidean"], ret_comp["per_query_euclidean"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"  Ret@1 cosine retention: {ret_cos_ci.value:.1f} ± {(ret_cos_ci.ci_upper - ret_cos_ci.ci_lower) / 2:.1f}%")
        print(f"  Ret@1 euclidean retention: {ret_euc_ci.value:.1f} ± {(ret_euc_ci.ci_upper - ret_euc_ci.ci_lower) / 2:.1f}%")

        all_results["retrieval"] = {
            "baseline_A_raw_mean": ret_A,
            "baseline_B_raw_dct4": ret_B,
            "baseline_C_abtt3_dct4": ret_C,
            "compressed": ret_comp,
            "retention_cosine_pct": ret_cos_ci.value,
            "retention_cosine": ret_cos_ci,
            "retention_euclidean_pct": ret_euc_ci.value,
            "retention_euclidean": ret_euc_ci,
        }

    # ==================================================================
    # 5. RETENTION SUMMARY
    # ==================================================================
    section("5. Retention Summary (Compressed / Baseline)")

    summary_rows = []

    def _fmt_ret_ci(r):
        hw = (r["retention"].ci_upper - r["retention"].ci_lower) / 2
        return f"{r['retention_pct']:.1f} ± {hw:.1f}%"

    if "ss3" in all_results:
        summary_rows.append(("SS3 Q3", all_results["ss3"]["retention_pct"], "vs raw 1024d"))
        print(f"  SS3 Q3:       {_fmt_ret_ci(all_results['ss3'])}  (vs raw 1024d)")

    if "ss8" in all_results:
        summary_rows.append(("SS8 Q8", all_results["ss8"]["retention_pct"], "vs raw 1024d"))
        print(f"  SS8 Q8:       {_fmt_ret_ci(all_results['ss8'])}  (vs raw 1024d)")

    if "disorder" in all_results:
        summary_rows.append(("Disorder rho", all_results["disorder"]["retention_pct"], "vs raw 1024d"))
        print(f"  Disorder rho: {_fmt_ret_ci(all_results['disorder'])}  (vs raw 1024d)")

    if "retrieval" in all_results:
        ret_cos = all_results["retrieval"]["retention_cosine_pct"]
        ret_euc = all_results["retrieval"]["retention_euclidean_pct"]
        hw_cos = (all_results["retrieval"]["retention_cosine"].ci_upper - all_results["retrieval"]["retention_cosine"].ci_lower) / 2
        hw_euc = (all_results["retrieval"]["retention_euclidean"].ci_upper - all_results["retrieval"]["retention_euclidean"].ci_lower) / 2
        summary_rows.append(("Ret@1 cosine", ret_cos, "vs Baseline C"))
        summary_rows.append(("Ret@1 euclidean", ret_euc, "vs Baseline C"))
        print(f"  Ret@1 cos:    {ret_cos:.1f} ± {hw_cos:.1f}%  (vs Baseline C: ABTT3+DCT4)")
        print(f"  Ret@1 euc:    {ret_euc:.1f} ± {hw_euc:.1f}%  (vs Baseline C: ABTT3+DCT4)")

    if summary_rows:
        valid = [r[1] for r in summary_rows if r[1] is not None]
        if valid:
            mean_ret = np.mean(valid)
            print(f"\n  Mean retention: {mean_ret:.1f}%")
            all_results["summary"] = {
                "tasks": [{"task": r[0], "retention_pct": r[1], "baseline": r[2]} for r in summary_rows],
                "mean_retention_pct": float(mean_ret),
            }
    else:
        print("  No benchmarks completed.")

    # ==================================================================
    # 6. SAVE RESULTS
    # ==================================================================
    section("6. Save Results")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "phase_a1_results.json"

    serializable = results_to_serializable(all_results)
    serializable["_meta"] = {
        "script": "run_phase_a1.py",
        "seeds": SEEDS,
        "bootstrap_n": BOOTSTRAP_N,
        "cv_folds": CV_FOLDS,
        "C_grid": C_GRID,
        "alpha_grid": ALPHA_GRID,
        "codec": "ABTT3 + RP768 + DCT K=4 (seed=42)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    print(f"\n  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
