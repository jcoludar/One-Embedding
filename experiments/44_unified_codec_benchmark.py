#!/usr/bin/env python3
"""Exp 44 — Unified Codec Benchmark Sweep.

Benchmarks all quantization types on 768d (and 1024d lossless) through the
Exp 43 rigorous framework using the NEW unified OneEmbeddingCodec from
src.one_embedding.codec_v2.

Configurations:
  - lossless:   d_out=1024, no RP, fp16 only (ABTT3 + fp16)
  - fp16-768:   d_out=768, ABTT3 + RP768 + fp16
  - int4-768:   d_out=768, ABTT3 + RP768 + int4 scalar
  - pq192-768:  d_out=768, ABTT3 + RP768 + PQ M=192
  - pq128-768:  d_out=768, ABTT3 + RP768 + PQ M=128
  - binary-768: d_out=768, ABTT3 + RP768 + binary (1-bit sign)

All benchmarks use:
  - BCa bootstrap CIs (B=10,000)
  - CV-tuned probes (GridSearchCV)
  - Averaged multi-seed predictions (3 seeds, Bouthillier et al. 2021)
  - Pooled Spearman rho for disorder (SETH/CAID standard)
  - Fair retrieval baselines (same DCT K=4 pooling)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "43_rigorous_benchmark"))

from config import (
    RAW_EMBEDDINGS, SPLITS, LABELS, METADATA, RESULTS_DIR,
    SEEDS, BOOTSTRAP_N, C_GRID, ALPHA_GRID, CV_FOLDS,
)
from runners.per_residue import run_ss3_benchmark, run_ss8_benchmark, run_disorder_benchmark
from runners.protein_level import compute_protein_vectors, run_retrieval_benchmark
from metrics.statistics import paired_bootstrap_retention, paired_cluster_bootstrap_retention
from rules import MetricResult

from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.utils.h5_store import load_residue_embeddings


# ── Configurations ────────────────────────────────────────────────────────

CONFIGS = [
    {"name": "lossless",   "d_out": 1024, "quantization": None,     "pq_m": None},
    {"name": "fp16-768",   "d_out": 768,  "quantization": None,     "pq_m": None},
    {"name": "int4-768",   "d_out": 768,  "quantization": "int4",   "pq_m": None},
    {"name": "pq192-768",  "d_out": 768,  "quantization": "pq",     "pq_m": 192},
    {"name": "pq128-768",  "d_out": 768,  "quantization": "pq",     "pq_m": 128},
    {"name": "binary-768", "d_out": 768,  "quantization": "binary", "pq_m": None},
]


# ── Helpers ───────────────────────────────────────────────────────────────

def load_split(path):
    with open(path) as f:
        return json.load(f)


def load_cb513_labels():
    from src.evaluation.per_residue_tasks import load_cb513_csv
    _, ss3, ss8, _ = load_cb513_csv(LABELS["cb513_csv"])
    return ss3, ss8


def load_chezod_labels():
    from src.evaluation.per_residue_tasks import load_chezod_seth
    _, disorder_scores, train_ids, test_ids = load_chezod_seth(LABELS["chezod_data_dir"])
    return disorder_scores, train_ids, test_ids


def load_metadata_with_families():
    import csv
    meta = []
    with open(METADATA["scope_5k"]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta.append(row)
    return meta


def metric_to_dict(m):
    """Convert MetricResult to JSON-serializable dict."""
    if isinstance(m, MetricResult):
        return {
            "value": m.value, "ci_lower": m.ci_lower, "ci_upper": m.ci_upper,
            "n": m.n, "seeds_mean": m.seeds_mean, "seeds_std": m.seeds_std,
            "ci_method": m.ci_method,
        }
    return m


def _pooled_spearman(cluster_data):
    """Compute pooled Spearman rho across all residues in cluster_data."""
    from scipy.stats import spearmanr
    all_true = np.concatenate([d["y_true"] for d in cluster_data])
    all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
    rho, _ = spearmanr(all_true, all_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def compress_embeddings(raw_embs: dict, codec) -> tuple[dict, dict]:
    """Encode + decode all proteins. Returns (per_residue_dict, protein_vec_dict)."""
    per_res = {}
    vecs = {}
    for pid, emb in raw_embs.items():
        encoded = codec.encode(emb)
        per_res[pid] = codec.decode_per_residue(encoded)
        vecs[pid] = encoded["protein_vec"].astype(np.float32)
    return per_res, vecs


def fmt_ret(mr_dict):
    """Format retention as XX.X+/-X.X%"""
    v = mr_dict["value"]
    hw = (mr_dict["ci_upper"] - mr_dict["ci_lower"]) / 2
    return f"{v:.1f}+/-{hw:.1f}%"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 44: Unified codec benchmark sweep (6 configs, rigorous framework)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 1 config with minimal bootstrap for testing",
    )
    args = parser.parse_args()

    configs = CONFIGS
    bootstrap_n = BOOTSTRAP_N

    if args.smoke_test:
        configs = [configs[0]]  # just lossless
        bootstrap_n = 100  # minimal

    print("=" * 70)
    print("Exp 44 — Unified Codec Benchmark Sweep")
    print("BCa CIs, CV-tuned probes, averaged seeds, pooled disorder rho")
    if args.smoke_test:
        print("*** SMOKE TEST MODE: 1 config, B=100 ***")
    print("=" * 70)
    print()

    t_total = time.time()

    # ── Load data ──
    print("Loading data...")
    raw_scope = load_residue_embeddings(RAW_EMBEDDINGS["prot_t5"])
    raw_cb513 = load_residue_embeddings(RAW_EMBEDDINGS["prot_t5_cb513"])
    raw_chezod = load_residue_embeddings(RAW_EMBEDDINGS["prot_t5_chezod"])

    ss3_labels, ss8_labels = load_cb513_labels()
    disorder_scores, dis_train_ids, dis_test_ids = load_chezod_labels()
    metadata = load_metadata_with_families()

    cb513_split = load_split(SPLITS["cb513"])
    cb_train = cb513_split["train_ids"]
    cb_test = cb513_split["test_ids"]

    scope_split = load_split(SPLITS["scope_5k"])
    scope_test = scope_split["test_ids"]
    scope_train_ids = scope_split["train_ids"]

    # Filter to available
    cb_train = [p for p in cb_train if p in raw_cb513 and p in ss3_labels]
    cb_test = [p for p in cb_test if p in raw_cb513 and p in ss3_labels]
    dis_train_ids = [p for p in dis_train_ids if p in raw_chezod and p in disorder_scores]
    dis_test_ids = [p for p in dis_test_ids if p in raw_chezod and p in disorder_scores]

    print(f"  SCOPe: {len(raw_scope)} proteins ({len(scope_train_ids)} train)")
    print(f"  CB513: {len(cb_train)} train, {len(cb_test)} test")
    print(f"  CheZOD: {len(dis_train_ids)} train, {len(dis_test_ids)} test")
    print()

    # ── Compute raw 1024d baselines (for retention calculation) ──
    print("Computing raw 1024d baselines...")

    raw_ss3 = run_ss3_benchmark(
        raw_cb513, ss3_labels, cb_train, cb_test,
        C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=bootstrap_n,
    )
    print(f"  Raw SS3 Q3: {raw_ss3['q3'].value:.4f}")

    raw_ss8 = run_ss8_benchmark(
        raw_cb513, ss8_labels, cb_train, cb_test,
        C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=bootstrap_n,
    )
    print(f"  Raw SS8 Q8: {raw_ss8['q8'].value:.4f}")

    raw_dis = run_disorder_benchmark(
        raw_chezod, disorder_scores, dis_train_ids, dis_test_ids,
        alpha_grid=ALPHA_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=bootstrap_n,
    )
    print(f"  Raw Disorder pooled rho: {raw_dis['pooled_spearman_rho'].value:.4f}")

    raw_ret_vecs = compute_protein_vectors(raw_scope, method="dct_k4")
    raw_ret = run_retrieval_benchmark(raw_ret_vecs, metadata, n_bootstrap=bootstrap_n)
    print(f"  Raw Ret@1 cosine: {raw_ret['ret1_cosine'].value:.4f}")
    print()

    # ── Benchmark each configuration ──
    all_results = {}

    for cfg in configs:
        name = cfg["name"]
        print("=" * 70)
        print(f"  CONFIG: {name}  (d_out={cfg['d_out']}, quant={cfg['quantization']}, pq_m={cfg['pq_m']})")
        print("=" * 70)
        t0 = time.time()

        # Create and fit codec
        codec = OneEmbeddingCodec(
            d_out=cfg["d_out"],
            quantization=cfg["quantization"],
            pq_m=cfg["pq_m"],
        )
        train_embs = {k: v for k, v in raw_scope.items() if k in set(scope_train_ids)}
        print(f"  Fitting codec on {len(train_embs)} train proteins...")
        codec.fit(train_embs)

        # Encode + decode CB513 for SS3/SS8
        cb513_decoded, _ = compress_embeddings(raw_cb513, codec)
        dim = next(iter(cb513_decoded.values())).shape[1]
        print(f"  Encoded CB513: {len(cb513_decoded)} proteins, dim={dim}")

        # Encode + decode CheZOD for disorder
        chezod_subset = {pid: emb for pid, emb in raw_chezod.items() if pid in disorder_scores}
        chezod_decoded, _ = compress_embeddings(chezod_subset, codec)

        # Encode SCOPe for retrieval (protein vectors)
        _, scope_vecs = compress_embeddings(raw_scope, codec)

        # ── SS3 ──
        print("  Running SS3...")
        comp_ss3 = run_ss3_benchmark(
            cb513_decoded, ss3_labels, cb_train, cb_test,
            C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
            n_bootstrap=bootstrap_n,
        )
        ss3_ret_ci = paired_bootstrap_retention(
            raw_ss3["per_protein_scores"], comp_ss3["per_protein_scores"],
            n_bootstrap=bootstrap_n, seed=SEEDS[0],
        )
        print(f"    Q3: {comp_ss3['q3'].value:.4f} (retention: {ss3_ret_ci.value:.1f} +/- {(ss3_ret_ci.ci_upper - ss3_ret_ci.ci_lower) / 2:.1f}%)")

        # ── SS8 ──
        print("  Running SS8...")
        comp_ss8 = run_ss8_benchmark(
            cb513_decoded, ss8_labels, cb_train, cb_test,
            C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
            n_bootstrap=bootstrap_n,
        )
        ss8_ret_ci = paired_bootstrap_retention(
            raw_ss8["per_protein_scores"], comp_ss8["per_protein_scores"],
            n_bootstrap=bootstrap_n, seed=SEEDS[0],
        )
        print(f"    Q8: {comp_ss8['q8'].value:.4f} (retention: {ss8_ret_ci.value:.1f} +/- {(ss8_ret_ci.ci_upper - ss8_ret_ci.ci_lower) / 2:.1f}%)")

        # ── Disorder ──
        print("  Running disorder...")
        comp_dis = run_disorder_benchmark(
            chezod_decoded, disorder_scores, dis_train_ids, dis_test_ids,
            alpha_grid=ALPHA_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
            n_bootstrap=bootstrap_n,
        )
        print(f"    Pooled rho: {comp_dis['pooled_spearman_rho'].value:.4f}")
        print(f"    AUC-ROC: {comp_dis['auc_roc'].value:.4f}")

        # Paired cluster bootstrap retention for disorder (pooled rho)
        dis_ret_ci = paired_cluster_bootstrap_retention(
            raw_dis["per_protein_predictions"], comp_dis["per_protein_predictions"],
            _pooled_spearman, n_bootstrap=bootstrap_n, seed=SEEDS[0],
        )
        print(f"    Disorder retention: {dis_ret_ci.value:.1f} +/- {(dis_ret_ci.ci_upper - dis_ret_ci.ci_lower) / 2:.1f}%")

        # ── Retrieval ──
        print("  Running retrieval...")
        comp_ret = run_retrieval_benchmark(scope_vecs, metadata, n_bootstrap=bootstrap_n)
        ret_ret_ci = paired_bootstrap_retention(
            raw_ret["per_query_cosine"], comp_ret["per_query_cosine"],
            n_bootstrap=bootstrap_n, seed=SEEDS[0],
        )
        print(f"    Ret@1 cosine: {comp_ret['ret1_cosine'].value:.4f} (retention: {ret_ret_ci.value:.1f} +/- {(ret_ret_ci.ci_upper - ret_ret_ci.ci_lower) / 2:.1f}%)")

        elapsed = time.time() - t0
        print(f"  Config {name} took {elapsed:.1f}s")

        all_results[name] = {
            "config": cfg,
            "ss3_q3": metric_to_dict(comp_ss3["q3"]),
            "ss3_retention": metric_to_dict(ss3_ret_ci),
            "ss8_q8": metric_to_dict(comp_ss8["q8"]),
            "ss8_retention": metric_to_dict(ss8_ret_ci),
            "disorder_pooled_rho": metric_to_dict(comp_dis["pooled_spearman_rho"]),
            "disorder_auc_roc": metric_to_dict(comp_dis["auc_roc"]),
            "disorder_retention": metric_to_dict(dis_ret_ci),
            "ret1_cosine": metric_to_dict(comp_ret["ret1_cosine"]),
            "ret1_retention": metric_to_dict(ret_ret_ci),
            "best_C_ss3": comp_ss3["best_C"],
            "best_C_ss8": comp_ss8["best_C"],
            "best_alpha_disorder": comp_dis["best_alpha"],
            "time_s": elapsed,
        }
        print()

    # ── Summary table ──
    config_names = [cfg["name"] for cfg in configs]

    print("=" * 70)
    print("SUMMARY: Exp 44 — Unified Codec Benchmarks (vs raw ProtT5 1024d)")
    print("=" * 70)
    print(f"{'Config':>12s}  {'SS3 Q3':>8s}  {'SS8 Q8':>8s}  {'Dis rho':>8s}  {'AUC':>6s}  {'Ret@1':>8s}  {'SS3 Ret':>14s}  {'SS8 Ret':>14s}  {'Dis Ret':>14s}  {'Ret Ret':>14s}")
    print("-" * 130)

    for name in config_names:
        if name not in all_results:
            continue
        r = all_results[name]
        print(f"{name:>12s}  "
              f"{r['ss3_q3']['value']:>8.4f}  "
              f"{r['ss8_q8']['value']:>8.4f}  "
              f"{r['disorder_pooled_rho']['value']:>8.4f}  "
              f"{r['disorder_auc_roc']['value']:>6.3f}  "
              f"{r['ret1_cosine']['value']:>8.4f}  "
              f"{fmt_ret(r['ss3_retention']):>14s}  "
              f"{fmt_ret(r['ss8_retention']):>14s}  "
              f"{fmt_ret(r['disorder_retention']):>14s}  "
              f"{fmt_ret(r['ret1_retention']):>14s}")

    print()
    print(f"Raw baselines: SS3={raw_ss3['q3'].value:.4f}, SS8={raw_ss8['q8'].value:.4f}, "
          f"Dis={raw_dis['pooled_spearman_rho'].value:.4f}, Ret@1={raw_ret['ret1_cosine'].value:.4f}")

    # ── Save results ──
    output = {
        "raw_baselines": {
            "ss3_q3": metric_to_dict(raw_ss3["q3"]),
            "ss8_q8": metric_to_dict(raw_ss8["q8"]),
            "disorder_pooled_rho": metric_to_dict(raw_dis["pooled_spearman_rho"]),
            "ret1_cosine": metric_to_dict(raw_ret["ret1_cosine"]),
        },
        "configs": all_results,
        "_meta": {
            "script": "44_unified_codec_benchmark.py",
            "methodology": "BCa bootstrap, CV-tuned probes, averaged 3-seed, pooled disorder rho",
            "seeds": SEEDS,
            "n_bootstrap": bootstrap_n,
            "C_grid": C_GRID,
            "alpha_grid": ALPHA_GRID,
            "smoke_test": args.smoke_test,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "exp44_unified_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
