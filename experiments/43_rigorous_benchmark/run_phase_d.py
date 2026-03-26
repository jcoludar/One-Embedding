#!/usr/bin/env python
"""Phase D: Ablation study + stress tests for the One Embedding codec.

Answers: which codec component contributes what, and does the codec
break on edge cases?

Codec pipeline: Raw (L, 1024) -> ABTT3 -> RP768 -> float16 -> DCT K=4.

Section 1 — Ablation Study
    Tests each component in isolation on SS3 (Q3) and Retrieval (Ret@1):
        Raw           — no processing, 1024d
        ABTT only     — remove top-3 PCs, 1024d
        RP only       — random projection, no ABTT, 768d
        ABTT + RP     — full pipeline, float32, 768d
        ABTT+RP+fp16  — shipped codec, 768d

Section 2 — Length Stress Test
    SCOPe 5K proteins binned by length.
    Retrieval (raw vs compressed) per bin.
    Check: does retention degrade with length?

Section 3 — Edge Cases
    Synthetic data: L=1, L=5, all-zero, large values, mixed-sign.
    Verify round-trip sanity (no NaN, finite, correct shape).

Section 4 — Summary
    Ablation table + length curve + edge case pass/fail.
    Results saved to RESULTS_DIR / "phase_d_results.json".

Usage:
    uv run python experiments/43_rigorous_benchmark/run_phase_d.py
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

# Runners (from experiment dir)
from runners.per_residue import run_ss3_benchmark
from runners.protein_level import (
    compute_protein_vectors,
    run_retrieval_benchmark,
)
from metrics.statistics import (
    bootstrap_ci,
    paired_bootstrap_retention,
)

# Project-level imports
from src.one_embedding.core.codec import Codec
from src.one_embedding.core.preprocessing import fit_abtt, apply_abtt
from src.one_embedding.core.projection import project
from src.one_embedding.preprocessing import (
    all_but_the_top,
    compute_corpus_stats,
)
from src.evaluation.per_residue_tasks import load_cb513_csv


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


def fmt_retention_ci(name: str, mr: MetricResult) -> str:
    return (
        f"  {name}: {mr.value:.1f}% "
        f"(95% CI: [{mr.ci_lower:.1f}%, {mr.ci_upper:.1f}%], n={mr.n})"
    )


def section(title: str) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*70}", flush=True)


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
# Serialization
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
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Ablation condition builders
# ---------------------------------------------------------------------------

def build_raw_condition(
    raw_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Raw — no processing at all. Per-residue stays 1024d."""
    return {pid: emb.astype(np.float32) for pid, emb in raw_embeddings.items()}


def build_abtt_only_condition(
    raw_embeddings: dict[str, np.ndarray],
    corpus_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """ABTT only — remove top-3 PCs, keep 1024d."""
    # Fit ABTT on the corpus (SCOPe 5K)
    residues = np.vstack(
        [np.asarray(v, dtype=np.float32) for v in corpus_embeddings.values()]
    )
    params = fit_abtt(residues, k=3, seed=42)

    result = {}
    for pid, emb in raw_embeddings.items():
        result[pid] = apply_abtt(emb, params)
    return result


def build_rp_only_condition(
    raw_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """RP only — random projection to 768d, no ABTT."""
    result = {}
    for pid, emb in raw_embeddings.items():
        result[pid] = project(emb, d_out=768, seed=42)
    return result


def build_abtt_rp_f32_condition(
    raw_embeddings: dict[str, np.ndarray],
    corpus_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """ABTT + RP — full pipeline in float32 (no fp16 quantization)."""
    # Fit ABTT on corpus
    residues = np.vstack(
        [np.asarray(v, dtype=np.float32) for v in corpus_embeddings.values()]
    )
    params = fit_abtt(residues, k=3, seed=42)

    result = {}
    for pid, emb in raw_embeddings.items():
        preprocessed = apply_abtt(emb, params)
        projected = project(preprocessed, d_out=768, seed=42)
        result[pid] = projected.astype(np.float32)
    return result


def build_abtt_rp_fp16_condition(
    raw_embeddings: dict[str, np.ndarray],
    corpus_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """ABTT + RP + fp16 — shipped codec (encode then cast back to fp32 for probes)."""
    codec = Codec(d_out=768, dct_k=4, seed=42)
    codec.fit(corpus_embeddings, k=3)

    result = {}
    for pid, emb in raw_embeddings.items():
        encoded = codec.encode(emb)
        # Codec returns fp16; cast back for downstream probes
        result[pid] = encoded["per_residue"].astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Ablation retrieval helper (builds protein vecs from per-residue)
# ---------------------------------------------------------------------------

def retrieval_for_condition(
    per_residue: dict[str, np.ndarray],
    metadata: list[dict],
    condition_name: str,
) -> dict:
    """Run retrieval benchmark for an ablation condition.

    Protein vectors are computed via DCT K=4 from the per-residue embeddings.
    """
    vecs = compute_protein_vectors(per_residue, method="dct_k4", dct_k=4)
    ret = run_retrieval_benchmark(
        vectors=vecs,
        metadata=metadata,
        label_key="family",
        n_bootstrap=BOOTSTRAP_N,
        seed=42,
    )
    return ret


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    all_results = {}

    # ==================================================================
    # Preflight: check data availability
    # ==================================================================
    cb513_emb_path = RAW_EMBEDDINGS["prot_t5_cb513"]
    cb513_split_path = SPLITS["cb513"]
    cb513_label_path = LABELS["cb513_csv"]
    scope_emb_path = RAW_EMBEDDINGS["prot_t5"]
    scope_meta_path = METADATA["scope_5k"]

    missing = []
    for name, path in [
        ("CB513 embeddings", cb513_emb_path),
        ("CB513 split", cb513_split_path),
        ("CB513 labels", cb513_label_path),
        ("SCOPe embeddings", scope_emb_path),
        ("SCOPe metadata", scope_meta_path),
    ]:
        if not path.exists():
            missing.append(f"  {name}: {path}")

    if missing:
        print("FATAL: Required data files not found:", flush=True)
        for m in missing:
            print(m, flush=True)
        sys.exit(1)

    # ==================================================================
    # Load shared data
    # ==================================================================
    section("Loading data")

    print("  Loading CB513 embeddings...", flush=True)
    raw_cb513 = load_h5_embeddings(cb513_emb_path)
    train_ids, test_ids = load_split(cb513_split_path)
    sequences, ss3_labels, ss8_labels, disorder_labels = load_cb513_csv(cb513_label_path)

    # Filter to IDs present in both embeddings and labels
    available_train = [pid for pid in train_ids if pid in raw_cb513 and pid in ss3_labels]
    available_test = [pid for pid in test_ids if pid in raw_cb513 and pid in ss3_labels]
    print(f"  CB513 — Train: {len(available_train)}, Test: {len(available_test)}", flush=True)

    print("  Loading SCOPe 5K embeddings...", flush=True)
    raw_scope = load_h5_embeddings(scope_emb_path)
    scope_metadata = load_scope_metadata(scope_meta_path)

    meta_ids = {m["id"] for m in scope_metadata}
    available_scope = [pid for pid in raw_scope if pid in meta_ids]
    print(f"  SCOPe — Proteins with embeddings + metadata: {len(available_scope)}", flush=True)

    # ==================================================================
    # SECTION 1: ABLATION STUDY
    # ==================================================================
    section("1. Ablation Study")
    print("  Testing each codec component in isolation.", flush=True)
    print("  Tasks: SS3 Q3 (CB513) + Retrieval Ret@1 cosine (SCOPe 5K)\n", flush=True)

    # Define ablation conditions
    # Each: (name, description, per_residue_dim, builder_fn)
    # We build per-residue dicts first, then run both benchmarks.

    conditions = [
        ("raw", "Raw 1024d (no processing)"),
        ("abtt_only", "ABTT only 1024d"),
        ("rp_only", "RP only 768d"),
        ("abtt_rp_f32", "ABTT + RP 768d (float32)"),
        ("abtt_rp_fp16", "ABTT + RP + fp16 768d (shipped codec)"),
    ]

    ablation_results = {}

    for cond_name, cond_desc in conditions:
        print(f"\n  --- Condition: {cond_desc} ---", flush=True)

        # Build per-residue embeddings for this condition
        if cond_name == "raw":
            per_res = build_raw_condition(raw_cb513)
            per_res_scope = build_raw_condition(raw_scope)
        elif cond_name == "abtt_only":
            per_res = build_abtt_only_condition(raw_cb513, raw_scope)
            per_res_scope = build_abtt_only_condition(raw_scope, raw_scope)
        elif cond_name == "rp_only":
            per_res = build_rp_only_condition(raw_cb513)
            per_res_scope = build_rp_only_condition(raw_scope)
        elif cond_name == "abtt_rp_f32":
            per_res = build_abtt_rp_f32_condition(raw_cb513, raw_scope)
            per_res_scope = build_abtt_rp_f32_condition(raw_scope, raw_scope)
        elif cond_name == "abtt_rp_fp16":
            per_res = build_abtt_rp_fp16_condition(raw_cb513, raw_scope)
            per_res_scope = build_abtt_rp_fp16_condition(raw_scope, raw_scope)
        else:
            raise ValueError(f"Unknown condition: {cond_name}")

        # --- SS3 benchmark ---
        print(f"  Running SS3 Q3...", flush=True)
        ss3_result = run_ss3_benchmark(
            embeddings=per_res,
            labels=ss3_labels,
            train_ids=available_train,
            test_ids=available_test,
            C_grid=C_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        q3_val = ss3_result["q3"].value
        print(fmt_metric(f"Q3 ({cond_name})", ss3_result["q3"]), flush=True)

        # --- Retrieval benchmark ---
        print(f"  Running Retrieval Ret@1 cosine...", flush=True)
        ret_result = retrieval_for_condition(
            per_res_scope, scope_metadata, cond_name,
        )
        ret1_cos = ret_result["ret1_cosine"].value
        print(fmt_metric(f"Ret@1 cosine ({cond_name})", ret_result["ret1_cosine"]), flush=True)

        ablation_results[cond_name] = {
            "description": cond_desc,
            "ss3_q3": ss3_result["q3"],
            "ss3_best_C": ss3_result["best_C"],
            "ret1_cosine": ret_result["ret1_cosine"],
            "ret1_euclidean": ret_result["ret1_euclidean"],
        }

    # --- Print ablation summary table ---
    print("\n" + "-" * 70, flush=True)
    print("  Ablation Results Summary", flush=True)
    print("-" * 70, flush=True)

    raw_q3 = ablation_results["raw"]["ss3_q3"].value
    raw_ret1 = ablation_results["raw"]["ret1_cosine"].value

    print(f"\n  {'Condition':<25s}  {'SS3 Q3':>8s}  {'delta pp':>9s}  "
          f"{'Ret@1 cos':>9s}  {'delta pp':>9s}", flush=True)
    print(f"  {'-'*25}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}", flush=True)

    for cond_name, cond_desc in conditions:
        res = ablation_results[cond_name]
        q3 = res["ss3_q3"].value
        ret1 = res["ret1_cosine"].value
        q3_delta = (q3 - raw_q3) * 100  # percentage points
        ret1_delta = (ret1 - raw_ret1) * 100

        label = cond_desc.split("(")[0].strip()  # Short label
        if cond_name == "raw":
            print(f"  {label:<25s}  {q3:>8.4f}  {'baseline':>9s}  "
                  f"{ret1:>8.4f}   {'baseline':>9s}", flush=True)
        else:
            q3_sign = "+" if q3_delta >= 0 else ""
            ret1_sign = "+" if ret1_delta >= 0 else ""
            print(f"  {label:<25s}  {q3:>8.4f}  {q3_sign}{q3_delta:>7.2f}pp  "
                  f"{ret1:>8.4f}   {ret1_sign}{ret1_delta:>7.2f}pp", flush=True)

    all_results["ablation"] = ablation_results

    # ==================================================================
    # SECTION 2: LENGTH STRESS TEST
    # ==================================================================
    section("2. Length Stress Test (SCOPe 5K)")
    print("  Binning proteins by length, checking retrieval retention per bin.", flush=True)

    # Get protein lengths from raw SCOPe embeddings
    protein_lengths = {pid: emb.shape[0] for pid, emb in raw_scope.items() if pid in meta_ids}

    length_bins = [
        ("short", 0, 100),
        ("medium", 100, 300),
        ("long", 300, 500),
        ("very_long", 500, float("inf")),
    ]

    # Build compressed SCOPe embeddings (full pipeline) + protein vecs
    print("  Compressing SCOPe embeddings with shipped codec...", flush=True)
    codec = Codec(d_out=768, dct_k=4, seed=42)
    codec.fit(raw_scope, k=3)

    comp_scope_per_res = {}
    for pid, emb in raw_scope.items():
        encoded = codec.encode(emb)
        comp_scope_per_res[pid] = encoded["per_residue"].astype(np.float32)

    # Compute protein vectors for raw and compressed
    raw_scope_vecs = compute_protein_vectors(raw_scope, method="dct_k4", dct_k=4)
    comp_scope_vecs = compute_protein_vectors(comp_scope_per_res, method="dct_k4", dct_k=4)

    length_results = {}

    for bin_name, lo, hi in length_bins:
        # Get protein IDs in this bin
        bin_ids = [
            pid for pid, L in protein_lengths.items()
            if lo <= L < hi
        ]
        n_proteins = len(bin_ids)

        if n_proteins < 10:
            print(f"\n  --- Bin '{bin_name}' (L in [{lo}, {hi})): "
                  f"SKIP ({n_proteins} proteins, need >= 10) ---", flush=True)
            length_results[bin_name] = {
                "range": f"[{lo}, {hi})",
                "n_proteins": n_proteins,
                "status": "skipped",
            }
            continue

        print(f"\n  --- Bin '{bin_name}' (L in [{lo}, {hi})): "
              f"{n_proteins} proteins ---", flush=True)

        # Filter metadata to this bin
        bin_meta = [m for m in scope_metadata if m["id"] in set(bin_ids)]

        # Raw retrieval on this bin
        raw_bin_vecs = {pid: raw_scope_vecs[pid] for pid in bin_ids if pid in raw_scope_vecs}
        ret_raw_bin = run_retrieval_benchmark(
            vectors=raw_bin_vecs,
            metadata=bin_meta,
            label_key="family",
            n_bootstrap=BOOTSTRAP_N,
            seed=42,
        )
        print(fmt_metric(f"Ret@1 cosine (raw, {bin_name})", ret_raw_bin["ret1_cosine"]), flush=True)

        # Compressed retrieval on this bin
        comp_bin_vecs = {pid: comp_scope_vecs[pid] for pid in bin_ids if pid in comp_scope_vecs}
        ret_comp_bin = run_retrieval_benchmark(
            vectors=comp_bin_vecs,
            metadata=bin_meta,
            label_key="family",
            n_bootstrap=BOOTSTRAP_N,
            seed=42,
        )
        print(fmt_metric(f"Ret@1 cosine (comp, {bin_name})", ret_comp_bin["ret1_cosine"]), flush=True)

        # Paired bootstrap retention
        from runners.protein_level import _retrieval_ret1
        raw_per_query = _retrieval_ret1(raw_bin_vecs, bin_meta, "family", "cosine")
        comp_per_query = _retrieval_ret1(comp_bin_vecs, bin_meta, "family", "cosine")

        if raw_per_query and comp_per_query:
            retention = paired_bootstrap_retention(
                raw_per_query,
                comp_per_query,
                n_bootstrap=BOOTSTRAP_N,
                seed=42,
            )
            print(fmt_retention_ci(f"Retention ({bin_name})", retention), flush=True)
        else:
            retention = MetricResult(value=0.0, ci_lower=0.0, ci_upper=0.0, n=0)
            print(f"  Retention ({bin_name}): could not compute (no queries)", flush=True)

        length_results[bin_name] = {
            "range": f"[{lo}, {hi})",
            "n_proteins": n_proteins,
            "ret1_raw_cosine": ret_raw_bin["ret1_cosine"],
            "ret1_comp_cosine": ret_comp_bin["ret1_cosine"],
            "retention": retention,
            "status": "ok",
        }

    # Print length summary table
    print("\n" + "-" * 70, flush=True)
    print("  Length Stress Test Summary", flush=True)
    print("-" * 70, flush=True)
    print(f"\n  {'Bin':<12s}  {'Range':<12s}  {'N':>5s}  {'Raw Ret@1':>9s}  "
          f"{'Comp Ret@1':>10s}  {'Retention':>10s}", flush=True)
    print(f"  {'-'*12}  {'-'*12}  {'-'*5}  {'-'*9}  {'-'*10}  {'-'*10}", flush=True)

    for bin_name, lo, hi in length_bins:
        res = length_results[bin_name]
        n = res["n_proteins"]
        if res["status"] == "skipped":
            print(f"  {bin_name:<12s}  [{lo},{hi}){'':>5s}  {n:>5d}  {'SKIPPED':>9s}  "
                  f"{'':>10s}  {'':>10s}", flush=True)
        else:
            raw_val = res["ret1_raw_cosine"].value
            comp_val = res["ret1_comp_cosine"].value
            ret_val = res["retention"].value
            print(f"  {bin_name:<12s}  [{lo},{hi}){'':>5s}  {n:>5d}  {raw_val:>9.4f}  "
                  f"{comp_val:>10.4f}  {ret_val:>9.1f}%", flush=True)

    all_results["length_stress"] = length_results

    # ==================================================================
    # SECTION 3: EDGE CASES
    # ==================================================================
    section("3. Edge Cases (Synthetic Data)")
    print("  Testing codec on synthetic edge-case embeddings.", flush=True)

    edge_codec = Codec(d_out=768, dct_k=4, seed=42)
    # Fit on a small synthetic corpus (just needs mean and PCs)
    # Use a minimal but valid corpus: 100 proteins of length 50
    rng = np.random.RandomState(42)
    synthetic_corpus = {
        f"synth_{i}": rng.randn(50, 1024).astype(np.float32)
        for i in range(100)
    }
    edge_codec.fit(synthetic_corpus, k=3)

    edge_cases = {
        "single_residue": np.random.RandomState(1).randn(1, 1024).astype(np.float32),
        "very_short_L5": np.random.RandomState(2).randn(5, 1024).astype(np.float32),
        "all_zero": np.zeros((20, 1024), dtype=np.float32),
        "very_large": np.random.RandomState(3).randn(20, 1024).astype(np.float32) * 1000,
        "mixed_sign": np.concatenate([
            np.abs(np.random.RandomState(4).randn(20, 512).astype(np.float32)),
            -np.abs(np.random.RandomState(5).randn(20, 512).astype(np.float32)),
        ], axis=1),
    }

    edge_results = {}

    for case_name, raw_emb in edge_cases.items():
        print(f"\n  --- Case: {case_name} (shape={raw_emb.shape}) ---", flush=True)
        checks = {}

        try:
            encoded = edge_codec.encode(raw_emb)
            per_res = encoded["per_residue"]
            prot_vec = encoded["protein_vec"]

            # Check 1: no crash
            checks["no_crash"] = True

            # Check 2: correct shapes
            expected_per_res_shape = (raw_emb.shape[0], 768)
            expected_prot_vec_dim = 4 * 768
            checks["per_res_shape_ok"] = (per_res.shape == expected_per_res_shape)
            checks["prot_vec_dim_ok"] = (prot_vec.shape[0] == expected_prot_vec_dim)

            # Check 3: no NaN
            per_res_f32 = per_res.astype(np.float32)
            prot_vec_f32 = prot_vec.astype(np.float32)
            checks["per_res_no_nan"] = bool(not np.any(np.isnan(per_res_f32)))
            checks["prot_vec_no_nan"] = bool(not np.any(np.isnan(prot_vec_f32)))

            # Check 4: all finite
            checks["per_res_finite"] = bool(np.all(np.isfinite(per_res_f32)))
            checks["prot_vec_finite"] = bool(np.all(np.isfinite(prot_vec_f32)))

            # Check 5: dtype is float16 (shipped default)
            checks["per_res_dtype_fp16"] = (per_res.dtype == np.float16)
            checks["prot_vec_dtype_fp16"] = (prot_vec.dtype == np.float16)

            # For non-zero input: check values are non-trivial
            if case_name != "all_zero":
                checks["per_res_nonzero"] = bool(np.any(per_res_f32 != 0))
                checks["prot_vec_nonzero"] = bool(np.any(prot_vec_f32 != 0))
            else:
                # All-zero input should produce all-zero output (linear pipeline)
                checks["per_res_all_zero"] = bool(np.allclose(per_res_f32, 0, atol=1e-6))
                checks["prot_vec_all_zero"] = bool(np.allclose(prot_vec_f32, 0, atol=1e-6))

            all_passed = all(checks.values())

        except Exception as e:
            checks["no_crash"] = False
            checks["error"] = str(e)
            all_passed = False

        # Report
        status = "PASS" if all_passed else "FAIL"
        print(f"  Result: {status}", flush=True)
        for check_name, check_val in checks.items():
            marker = "OK" if check_val else "FAIL"
            print(f"    {check_name}: {marker}", flush=True)

        edge_results[case_name] = {
            "shape": list(raw_emb.shape),
            "checks": checks,
            "passed": all_passed,
        }

    all_results["edge_cases"] = edge_results

    # ==================================================================
    # SECTION 4: SUMMARY
    # ==================================================================
    section("4. Summary")

    # --- Ablation summary ---
    print("\n  Ablation Study:", flush=True)
    print(f"  {'Condition':<25s}  {'SS3 Q3':>8s}  {'Ret@1 cos':>9s}", flush=True)
    print(f"  {'-'*25}  {'-'*8}  {'-'*9}", flush=True)
    for cond_name, cond_desc in conditions:
        res = ablation_results[cond_name]
        q3 = res["ss3_q3"].value
        ret1 = res["ret1_cosine"].value
        label = cond_desc.split("(")[0].strip()
        print(f"  {label:<25s}  {q3:>8.4f}  {ret1:>9.4f}", flush=True)

    # Component contributions (deltas vs raw)
    print("\n  Component Contributions (delta vs raw):", flush=True)
    raw_q3 = ablation_results["raw"]["ss3_q3"].value
    raw_ret1 = ablation_results["raw"]["ret1_cosine"].value

    abtt_q3_delta = (ablation_results["abtt_only"]["ss3_q3"].value - raw_q3) * 100
    abtt_ret1_delta = (ablation_results["abtt_only"]["ret1_cosine"].value - raw_ret1) * 100
    print(f"    ABTT alone:   SS3 {abtt_q3_delta:+.2f}pp, Ret@1 {abtt_ret1_delta:+.2f}pp", flush=True)

    rp_q3_delta = (ablation_results["rp_only"]["ss3_q3"].value - raw_q3) * 100
    rp_ret1_delta = (ablation_results["rp_only"]["ret1_cosine"].value - raw_ret1) * 100
    print(f"    RP alone:     SS3 {rp_q3_delta:+.2f}pp, Ret@1 {rp_ret1_delta:+.2f}pp", flush=True)

    combined_q3_delta = (ablation_results["abtt_rp_f32"]["ss3_q3"].value - raw_q3) * 100
    combined_ret1_delta = (ablation_results["abtt_rp_f32"]["ret1_cosine"].value - raw_ret1) * 100
    print(f"    ABTT+RP:      SS3 {combined_q3_delta:+.2f}pp, Ret@1 {combined_ret1_delta:+.2f}pp", flush=True)

    fp16_q3_delta = (ablation_results["abtt_rp_fp16"]["ss3_q3"].value -
                     ablation_results["abtt_rp_f32"]["ss3_q3"].value) * 100
    fp16_ret1_delta = (ablation_results["abtt_rp_fp16"]["ret1_cosine"].value -
                       ablation_results["abtt_rp_f32"]["ret1_cosine"].value) * 100
    print(f"    fp16 effect:  SS3 {fp16_q3_delta:+.2f}pp, Ret@1 {fp16_ret1_delta:+.2f}pp", flush=True)

    # --- Length stress summary ---
    print("\n  Length Stress Test:", flush=True)
    degradation_detected = False
    prev_retention = None
    for bin_name, lo, hi in length_bins:
        res = length_results[bin_name]
        if res["status"] == "ok":
            ret_val = res["retention"].value
            print(f"    {bin_name}: {ret_val:.1f}% retention", flush=True)
            if prev_retention is not None and ret_val < prev_retention - 5.0:
                degradation_detected = True
            prev_retention = ret_val
        else:
            print(f"    {bin_name}: skipped", flush=True)

    if degradation_detected:
        print("  WARNING: Retention degrades significantly with protein length.", flush=True)
    else:
        print("  OK: No significant length-dependent retention degradation detected.", flush=True)

    # --- Edge case summary ---
    print("\n  Edge Cases:", flush=True)
    n_passed = sum(1 for v in edge_results.values() if v["passed"])
    n_total = len(edge_results)
    for case_name, res in edge_results.items():
        status = "PASS" if res["passed"] else "FAIL"
        print(f"    {case_name}: {status}", flush=True)
    print(f"  Overall: {n_passed}/{n_total} passed", flush=True)

    all_results["summary"] = {
        "ablation_conditions": len(conditions),
        "length_bins": len(length_bins),
        "edge_cases_passed": n_passed,
        "edge_cases_total": n_total,
        "length_degradation_detected": degradation_detected,
    }

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
    section("Save Results")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "phase_d_results.json"

    serializable = results_to_serializable(all_results)
    serializable["_meta"] = {
        "script": "run_phase_d.py",
        "phase": "D",
        "description": "Ablation study + length stress test + edge cases",
        "seeds": SEEDS,
        "bootstrap_n": BOOTSTRAP_N,
        "cv_folds": CV_FOLDS,
        "C_grid": C_GRID,
        "codec": "ABTT3 + RP768 + DCT K=4 (seed=42)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}", flush=True)
    print(f"\n  Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
