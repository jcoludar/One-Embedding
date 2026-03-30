#!/usr/bin/env python3
"""V2 Extreme Compression — Rigorous Re-validation.

Runs all V2 codec modes (full, balanced, compact, micro, binary) through
the Exp 43 rigorous benchmark framework with:
- BCa bootstrap CIs
- CV-tuned probes (GridSearchCV)
- Averaged multi-seed predictions
- Pooled Spearman rho for disorder (SETH/CAID standard)
- Fair retrieval baselines

This replaces the Exp 34 results which used hardcoded C=1.0 probes.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# V2 mode configs (512d for historical reproducibility with Exp 34 / Exp 43 V2 tiers)
V2_CONFIGS = {
    "full":     {"d_out": 512, "quantization": "int4",   "pq_m": None, "desc": "int4 per-residue (V1 compatible)"},
    "balanced": {"d_out": 512, "quantization": "pq",     "pq_m": 128,  "desc": "PQ M=128"},
    "compact":  {"d_out": 512, "quantization": "pq",     "pq_m": 64,   "desc": "PQ M=64"},
    "micro":    {"d_out": 512, "quantization": "pq",     "pq_m": 32,   "desc": "PQ M=32"},
    "binary":   {"d_out": 512, "quantization": "binary", "pq_m": None, "desc": "1-bit sign quantization"},
}


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


def main():
    print("=" * 70)
    print("V2 Extreme Compression — Rigorous Re-validation")
    print("BCa CIs, CV-tuned probes, averaged seeds, pooled disorder rho")
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

    # Filter to available
    cb_train = [p for p in cb_train if p in raw_cb513 and p in ss3_labels]
    cb_test = [p for p in cb_test if p in raw_cb513 and p in ss3_labels]
    dis_train_ids = [p for p in dis_train_ids if p in raw_chezod and p in disorder_scores]
    dis_test_ids = [p for p in dis_test_ids if p in raw_chezod and p in disorder_scores]

    print(f"  SCOPe: {len(raw_scope)} proteins")
    print(f"  CB513: {len(cb_train)} train, {len(cb_test)} test")
    print(f"  CheZOD: {len(dis_train_ids)} train, {len(dis_test_ids)} test")
    print()

    # ── Also get raw 1024d baselines (for retention calculation) ──
    print("Computing raw 1024d baselines...")

    raw_ss3 = run_ss3_benchmark(
        raw_cb513, ss3_labels, cb_train, cb_test,
        C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(f"  Raw SS3 Q3: {raw_ss3['q3'].value:.4f}")

    raw_ss8 = run_ss8_benchmark(
        raw_cb513, ss8_labels, cb_train, cb_test,
        C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(f"  Raw SS8 Q8: {raw_ss8['q8'].value:.4f}")

    raw_dis = run_disorder_benchmark(
        raw_chezod, disorder_scores, dis_train_ids, dis_test_ids,
        alpha_grid=ALPHA_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(f"  Raw Disorder pooled rho: {raw_dis['pooled_spearman_rho'].value:.4f}")

    raw_ret_vecs = compute_protein_vectors(raw_scope, method="dct_k4")
    raw_ret = run_retrieval_benchmark(raw_ret_vecs, metadata, n_bootstrap=BOOTSTRAP_N)
    print(f"  Raw Ret@1 cosine: {raw_ret['ret1_cosine'].value:.4f}")
    print()

    # ── Benchmark each V2 mode ──
    modes = ["full", "balanced", "compact", "micro", "binary"]
    all_results = {}

    for mode in modes:
        print("=" * 70)
        print(f"  MODE: {mode} — {V2_CONFIGS[mode]['desc']}")
        print("=" * 70)
        t0 = time.time()

        # Fit codec on SCOPe train
        cfg = V2_CONFIGS[mode]
        codec = OneEmbeddingCodec(d_out=cfg["d_out"], quantization=cfg["quantization"], pq_m=cfg["pq_m"])
        scope_train_ids = scope_split["train_ids"]
        train_embs = {k: v for k, v in raw_scope.items() if k in set(scope_train_ids)}
        print(f"  Fitting codebook on {len(train_embs)} train proteins...")
        codec.fit(train_embs)

        # Encode + decode CB513
        cb513_decoded = {}
        for pid, emb in raw_cb513.items():
            enc = codec.encode(emb)
            cb513_decoded[pid] = codec.decode_per_residue(enc)
        print(f"  Encoded CB513: {len(cb513_decoded)} proteins, dim={next(iter(cb513_decoded.values())).shape[1]}")

        # Encode + decode CheZOD
        chezod_decoded = {}
        for pid, emb in raw_chezod.items():
            if pid in disorder_scores:
                enc = codec.encode(emb)
                chezod_decoded[pid] = codec.decode_per_residue(enc)

        # Encode SCOPe for retrieval (protein vectors)
        scope_vecs = {}
        for pid, emb in raw_scope.items():
            enc = codec.encode(emb)
            scope_vecs[pid] = enc["protein_vec"].astype(np.float32)

        # ── SS3 ──
        print("  Running SS3...")
        v2_ss3 = run_ss3_benchmark(
            cb513_decoded, ss3_labels, cb_train, cb_test,
            C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        ss3_ret_ci = paired_bootstrap_retention(
            raw_ss3["per_protein_scores"], v2_ss3["per_protein_scores"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"    Q3: {v2_ss3['q3'].value:.4f} (retention: {ss3_ret_ci.value:.1f} ± {(ss3_ret_ci.ci_upper - ss3_ret_ci.ci_lower) / 2:.1f}%)")

        # ── SS8 ──
        print("  Running SS8...")
        v2_ss8 = run_ss8_benchmark(
            cb513_decoded, ss8_labels, cb_train, cb_test,
            C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        ss8_ret_ci = paired_bootstrap_retention(
            raw_ss8["per_protein_scores"], v2_ss8["per_protein_scores"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"    Q8: {v2_ss8['q8'].value:.4f} (retention: {ss8_ret_ci.value:.1f} ± {(ss8_ret_ci.ci_upper - ss8_ret_ci.ci_lower) / 2:.1f}%)")

        # ── Disorder ──
        print("  Running disorder...")
        v2_dis = run_disorder_benchmark(
            chezod_decoded, disorder_scores, dis_train_ids, dis_test_ids,
            alpha_grid=ALPHA_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
        )
        print(f"    Pooled rho: {v2_dis['pooled_spearman_rho'].value:.4f}")
        print(f"    AUC-ROC: {v2_dis['auc_roc'].value:.4f}")

        # Paired cluster bootstrap retention for disorder (pooled rho)
        dis_ret_ci = paired_cluster_bootstrap_retention(
            raw_dis["per_protein_predictions"], v2_dis["per_protein_predictions"],
            _pooled_spearman, n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"    Disorder retention: {dis_ret_ci.value:.1f} ± {(dis_ret_ci.ci_upper - dis_ret_ci.ci_lower) / 2:.1f}%")

        # ── Retrieval ──
        print("  Running retrieval...")
        v2_ret = run_retrieval_benchmark(scope_vecs, metadata, n_bootstrap=BOOTSTRAP_N)
        ret_ret_ci = paired_bootstrap_retention(
            raw_ret["per_query_cosine"], v2_ret["per_query_cosine"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"    Ret@1 cosine: {v2_ret['ret1_cosine'].value:.4f} (retention: {ret_ret_ci.value:.1f} ± {(ret_ret_ci.ci_upper - ret_ret_ci.ci_lower) / 2:.1f}%)")

        elapsed = time.time() - t0
        print(f"  Mode {mode} took {elapsed:.1f}s")

        all_results[mode] = {
            "ss3_q3": metric_to_dict(v2_ss3["q3"]),
            "ss3_retention": metric_to_dict(ss3_ret_ci),
            "ss8_q8": metric_to_dict(v2_ss8["q8"]),
            "ss8_retention": metric_to_dict(ss8_ret_ci),
            "disorder_pooled_rho": metric_to_dict(v2_dis["pooled_spearman_rho"]),
            "disorder_auc_roc": metric_to_dict(v2_dis["auc_roc"]),
            "disorder_retention": metric_to_dict(dis_ret_ci),
            "ret1_cosine": metric_to_dict(v2_ret["ret1_cosine"]),
            "ret1_retention": metric_to_dict(ret_ret_ci),
            "best_C_ss3": v2_ss3["best_C"],
            "best_C_ss8": v2_ss8["best_C"],
            "best_alpha_disorder": v2_dis["best_alpha"],
            "time_s": elapsed,
        }
        print()

    # ── Summary table ──
    def fmt_ret(mr_dict):
        """Format retention as XX.X±X.X%"""
        v = mr_dict["value"]
        hw = (mr_dict["ci_upper"] - mr_dict["ci_lower"]) / 2
        return f"{v:.1f}±{hw:.1f}%"

    print("=" * 70)
    print("SUMMARY: V2 Modes — Rigorous Retention (vs raw ProtT5 1024d)")
    print("=" * 70)
    print(f"{'Mode':>10s}  {'SS3 Q3':>8s}  {'SS8 Q8':>8s}  {'Dis ρ':>8s}  {'AUC':>6s}  {'Ret@1':>8s}  {'SS3 Ret':>14s}  {'SS8 Ret':>14s}  {'Dis Ret':>14s}  {'Ret Ret':>14s}")
    print("-" * 120)

    for mode in modes:
        r = all_results[mode]
        print(f"{mode:>10s}  "
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
        "modes": all_results,
        "_meta": {
            "script": "run_v2_validation.py",
            "methodology": "BCa bootstrap, CV-tuned probes, averaged 3-seed, pooled disorder rho",
            "seeds": SEEDS,
            "n_bootstrap": BOOTSTRAP_N,
            "C_grid": C_GRID,
            "alpha_grid": ALPHA_GRID,
        },
    }

    results_path = RESULTS_DIR / "v2_rigorous_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
