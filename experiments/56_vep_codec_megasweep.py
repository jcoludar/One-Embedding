#!/usr/bin/env python3
"""Experiment 56: VEP codec mega-sweep — ABTT, dimensionality, alt quantization.

Extends Exp 55 with 11 additional codec arms over three axes (ABTT-k,
d_out, quantization variant) on the same cached ProteinGym diversity +
ClinVar embeddings. Headline questions:

  * Does ABTT (top-PC removal) hurt VEP retention the way Exp 45 showed
    it hurts disorder? — binary_896_abtt{1,3,8} answer this.
  * Does dropping the random projection help binary at the lossless input
    dim? — binary_1024 (no RP) and binary_1024_abtt3 answer this.
  * How does retention scale into more aggressive compression? — binary_512
    (64×), pq64_896 (64×), int2_896 (18×) answer this.
  * Does the polar-quant variant rejected for disorder (Exp 51) help VEP
    at all? — binary_magnitude_896 answers this.
  * Does ABTT × quantization interact non-trivially? — fp16_896_abtt3 and
    int4_896_abtt3 isolate it.

Re-uses the Exp 55 cached embeddings (symlinked into this worktree) and
the same Ridge probe + ClinVar + RNS sections. Lossless 1024d is re-run
here so each new arm has paired predictions for its retention CI.

Three sections (identical to Exp 55):
  1. DMS retention  — Ridge probe Spearman ρ per assay per codec; paired
                      BCa bootstrap retention CIs vs lossless baseline.
  2. ClinVar AUC    — Zero-shot cosine-distance scoring; AUC per codec.
  3. RNS ride-along — Pearson r between RNS and VEP ρ (raw + binary_896).

Usage:
    uv run python experiments/56_vep_codec_megasweep.py --smoke-test --skip-clinvar --skip-rns
    uv run python experiments/56_vep_codec_megasweep.py --skip-clinvar
    uv run python experiments/56_vep_codec_megasweep.py  # full sweep (~4h)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

from src.one_embedding.vep import (
    CodecSpec,
    ProbeResult,
    bootstrap_ci_paired,
    bootstrap_ci_pearson,
    build_variant_features,
    clinvar_auc,
    compute_rns_for_assays,
    evaluate_assay_across_codecs,
    evaluate_assay_streaming,
    fit_evaluate_ridge_probe,
    score_clinvar_zeroshot,
)
from src.one_embedding.codec_v2 import OneEmbeddingCodec

DATA = ROOT / "data"
EMB_DIR = DATA / "residue_embeddings"
OUT_DMS_H5 = EMB_DIR / "prot_t5_xl_proteingym_diversity.h5"
OUT_CLINVAR_H5 = EMB_DIR / "prot_t5_xl_proteingym_clinvar.h5"
RESULTS_DIR = DATA / "benchmarks" / "rigorous_v1"

SEEDS = [42, 123, 456]
BOOTSTRAP_N = 10_000

# ── Codec specifications ───────────────────────────────────────────────────
# Lossless 1024d is re-run here as the paired-retention baseline. Every new
# arm changes one knob relative to the Exp 55 default (binary 896 abtt=0).

ALL_CODEC_SPECS: dict[str, CodecSpec] = {
    # Baseline (re-run for paired retention)
    "lossless":             CodecSpec(name="lossless",             d_out=1024, quantization=None),

    # ABTT-k axis on binary 896 (Exp 45 showed ABTT3 destroys disorder; does it hurt VEP too?)
    "binary_896_abtt1":     CodecSpec(name="binary_896_abtt1",     d_out=896,  quantization="binary",           abtt_k=1),
    "binary_896_abtt3":     CodecSpec(name="binary_896_abtt3",     d_out=896,  quantization="binary",           abtt_k=3),
    "binary_896_abtt8":     CodecSpec(name="binary_896_abtt8",     d_out=896,  quantization="binary",           abtt_k=8),

    # Dimensionality axis on binary
    "binary_1024":          CodecSpec(name="binary_1024",          d_out=1024, quantization="binary"),
    "binary_1024_abtt3":    CodecSpec(name="binary_1024_abtt3",    d_out=1024, quantization="binary",           abtt_k=3),
    "binary_512":           CodecSpec(name="binary_512",           d_out=512,  quantization="binary"),

    # Alt quantization variants
    "binary_magnitude_896": CodecSpec(name="binary_magnitude_896", d_out=896,  quantization="binary_magnitude"),
    "pq128_896":            CodecSpec(name="pq128_896",            d_out=896,  quantization="pq",                pq_m=128),
    "pq64_896":             CodecSpec(name="pq64_896",             d_out=896,  quantization="pq",                pq_m=64),
    "int2_896":             CodecSpec(name="int2_896",             d_out=896,  quantization="int2"),

    # ABTT × quantization interaction
    "fp16_896_abtt3":       CodecSpec(name="fp16_896_abtt3",       d_out=896,  quantization=None,                abtt_k=3),
    "int4_896_abtt3":       CodecSpec(name="int4_896_abtt3",       d_out=896,  quantization="int4",              abtt_k=3),
}

DEFAULT_CODEC_NAMES = list(ALL_CODEC_SPECS.keys())
SMOKE_CODEC_NAMES = ["lossless", "binary_896_abtt3", "int2_896"]


# ── H5 loader ─────────────────────────────────────────────────────────────

def _load_assay_from_h5(hf: h5py.File, assay_id: str) -> dict | None:
    """Load one assay group from an open H5 file.

    Returns dict with keys:
        wt_emb: (L, 1024) float32
        wt_sequence: str
        variants: list of dicts with mut_pos, mut_emb, score
    Returns None if the group is malformed or has no variants.
    """
    if assay_id not in hf:
        return None
    grp = hf[assay_id]
    if "wt" not in grp or "variant_meta" not in grp.attrs:
        return None
    try:
        meta_list = json.loads(grp.attrs["variant_meta"])
    except (json.JSONDecodeError, KeyError):
        return None
    if not meta_list:
        print(f"  [SKIP] {assay_id}: no variant metadata", flush=True)
        return None

    wt_emb = grp["wt"][:].astype(np.float32)
    wt_sequence = grp.attrs.get("wt_sequence", "")

    variants = []
    for m in meta_list:
        v_key = f"v_{m['idx']}"
        if v_key not in grp:
            continue
        if m.get("score") is None:
            continue
        mut_emb = grp[v_key][:].astype(np.float32)
        variants.append({
            "mut_pos": int(m["mut_pos"]),
            "mut_emb": mut_emb,
            "score": float(m["score"]),
        })
    if not variants:
        return None
    return {"wt_emb": wt_emb, "wt_sequence": wt_sequence, "variants": variants}


def _load_clinvar_assay_from_h5(hf: h5py.File, pid: str) -> dict | None:
    """Load one ClinVar parent-protein group from an open H5 file.

    Returns dict with keys:
        wt_emb: (L, 1024) float32
        variants: list of dicts with mut_pos, mut_emb, label (0/1)
    Returns None if malformed or no labelled variants.
    """
    if pid not in hf:
        return None
    grp = hf[pid]
    if "wt" not in grp or "variant_meta" not in grp.attrs:
        return None
    try:
        meta_list = json.loads(grp.attrs["variant_meta"])
    except (json.JSONDecodeError, KeyError):
        return None
    if not meta_list:
        return None

    wt_emb = grp["wt"][:].astype(np.float32)

    variants = []
    for m in meta_list:
        v_key = f"v_{m['idx']}"
        if v_key not in grp:
            continue
        if m.get("label") is None:
            continue
        mut_emb = grp[v_key][:].astype(np.float32)
        variants.append({
            "mut_pos": int(m["mut_pos"]),
            "mut_emb": mut_emb,
            "label": int(m["label"]),
        })
    if not variants:
        return None
    return {"wt_emb": wt_emb, "variants": variants}


# ── Section 1: DMS retention ───────────────────────────────────────────────

def run_dms_retention(
    codec_names: list[str],
    seeds: list[int] = SEEDS,
    n_boot: int = BOOTSTRAP_N,
) -> tuple[dict, dict[str, np.ndarray], dict[str, str]]:
    """Run DMS probe benchmarks across codecs.

    Returns:
        results: dict keyed by codec name with per-assay ρ and retention CIs
        wt_embs: {assay_id: (L, 1024) float32} for RNS ride-along
        wt_seqs: {assay_id: str} for RNS ride-along
    """
    if not OUT_DMS_H5.exists():
        raise FileNotFoundError(
            f"DMS embedding H5 not found: {OUT_DMS_H5}\n"
            "Run: uv run python experiments/55b_extract_variant_embeddings.py"
        )

    codecs = [ALL_CODEC_SPECS[n] for n in codec_names]

    print(f"\n{'='*70}")
    print("DMS RETENTION")
    print(f"  Codecs: {codec_names}")
    print(f"  H5: {OUT_DMS_H5}")
    print(f"{'='*70}")

    # First pass: load just WT embeddings (small) to build fit_corpus + RNS-side caches.
    # Variants are loaded streaming, one assay at a time, to keep peak RAM bounded.
    wt_embs: dict[str, np.ndarray] = {}
    wt_seqs: dict[str, str] = {}
    assay_variant_counts: dict[str, int] = {}

    with h5py.File(OUT_DMS_H5, "r") as hf:
        assay_ids = sorted(hf.keys())
        print(f"\nAssays in H5: {len(assay_ids)}")
        for assay_id in assay_ids:
            grp = hf[assay_id]
            if "wt" not in grp or "variant_meta" not in grp.attrs:
                print(f"  [SKIP] {assay_id}: malformed group", flush=True)
                continue
            try:
                meta_list = json.loads(grp.attrs["variant_meta"])
            except (json.JSONDecodeError, KeyError):
                print(f"  [SKIP] {assay_id}: bad variant_meta", flush=True)
                continue
            if not meta_list:
                print(f"  [SKIP] {assay_id}: empty variant_meta", flush=True)
                continue
            wt_embs[assay_id] = grp["wt"][:].astype(np.float32)
            wt_seqs[assay_id] = grp.attrs.get("wt_sequence", "")
            assay_variant_counts[assay_id] = len(meta_list)
            print(f"  {assay_id}: L={wt_embs[assay_id].shape[0]}, n_variants={len(meta_list)}", flush=True)

    if not wt_embs:
        raise RuntimeError("No valid assays loaded from DMS H5.")

    # fit_corpus = WT embeddings only (small; centering stats and PQ codebook)
    fit_corpus = wt_embs  # alias — same dict

    # Evaluate all codecs across all assays — streaming variants per assay
    per_codec_rho: dict[str, dict[str, float]] = {n: {} for n in codec_names}

    for assay_id in sorted(wt_embs.keys()):
        n_var = assay_variant_counts[assay_id]
        print(f"\n  [{assay_id}]  n={n_var}", flush=True)
        t0 = time.time()
        # Streaming pattern: keep the H5 open across this assay's codec sweep,
        # define a closure that lazy-loads one variant at a time.
        try:
            with h5py.File(OUT_DMS_H5, "r") as hf:
                grp = hf[assay_id]
                meta_list = json.loads(grp.attrs["variant_meta"])
                valid = [
                    m for m in meta_list
                    if m.get("score") is not None and f"v_{m['idx']}" in grp
                ]
                if not valid:
                    print(f"    [SKIP] no scored variants", flush=True)
                    continue
                mut_positions = [int(m["mut_pos"]) for m in valid]
                scores = np.array([float(m["score"]) for m in valid], dtype=np.float32)

                def _loader(i: int, _grp=grp, _valid=valid) -> np.ndarray:
                    return _grp[f"v_{_valid[i]['idx']}"][:].astype(np.float32)

                results = evaluate_assay_streaming(
                    wt_emb=wt_embs[assay_id],
                    variant_loader=_loader,
                    n_variants=len(valid),
                    mut_positions=mut_positions,
                    scores=scores,
                    codecs=codecs,
                    fit_corpus=fit_corpus,
                    seeds=seeds,
                )
            for codec_name, probe_result in results.items():
                per_codec_rho[codec_name][assay_id] = probe_result.spearman_rho
                print(f"    {codec_name}: ρ={probe_result.spearman_rho:.3f}", flush=True)
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            import traceback; traceback.print_exc()
        import gc; gc.collect()
        print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # Aggregate: mean ρ per codec + paired retention CIs vs lossless
    lossless_name = "lossless"
    if lossless_name not in codec_names:
        lossless_name = codec_names[0]

    lossless_assay_rhos = per_codec_rho[lossless_name]
    shared_assays = sorted(lossless_assay_rhos.keys())

    codec_summaries: dict[str, dict] = {}
    for codec_name in codec_names:
        assay_rhos = per_codec_rho[codec_name]
        valid_assays = [a for a in shared_assays if a in assay_rhos]
        if not valid_assays:
            codec_summaries[codec_name] = {"error": "no valid assays"}
            continue

        rho_vals = np.array([assay_rhos[a] for a in valid_assays])
        lossless_vals = np.array([lossless_assay_rhos[a] for a in valid_assays])

        mean_rho = float(rho_vals.mean())
        retention = bootstrap_ci_paired(lossless_vals, rho_vals, n_boot=n_boot, seed=42)

        codec_summaries[codec_name] = {
            "mean_spearman_rho": mean_rho,
            "per_assay_rho": {a: float(assay_rhos[a]) for a in valid_assays},
            "retention_pct": retention["retention_pct"],
            "retention_ci_low": retention["ci_low"],
            "retention_ci_high": retention["ci_high"],
            "n_assays": len(valid_assays),
        }

    results_out = {
        "codecs": codec_summaries,
        "assay_ids": shared_assays,
    }

    # Print summary table
    print(f"\n{'─'*60}")
    print(f"{'Codec':<16} {'Mean ρ':>8} {'Retention':>12} {'CI':>18}")
    print(f"{'─'*60}")
    for cname, s in codec_summaries.items():
        if "error" in s:
            print(f"  {cname:<16} ERROR")
            continue
        half = (s["retention_ci_high"] - s["retention_ci_low"]) / 2
        print(f"  {cname:<16} {s['mean_spearman_rho']:>8.3f} "
              f"{s['retention_pct']:>10.1f}% "
              f"± {half:.1f}%")

    return results_out, wt_embs, wt_seqs


# ── Section 2: ClinVar AUC ─────────────────────────────────────────────────

def run_clinvar(
    codec_names: list[str],
    fit_corpus: dict[str, np.ndarray],
) -> dict:
    """Zero-shot ClinVar AUC per codec.

    Uses cosine-distance at the mutation site as pathogenicity score.
    """
    if not OUT_CLINVAR_H5.exists():
        print(f"\n[SKIP ClinVar] H5 not found: {OUT_CLINVAR_H5}", flush=True)
        return {"skipped": True, "reason": "H5 not found"}

    print(f"\n{'='*70}")
    print("CLINVAR AUC")
    print(f"  Codecs: {codec_names}")
    print(f"{'='*70}")

    # First pass: preload only WT embeddings + per-protein variant metadata.
    # Variants are streamed from H5 during the codec loop below.
    cv_wt_embs: dict[str, np.ndarray] = {}
    cv_meta: dict[str, list[dict]] = {}
    with h5py.File(OUT_CLINVAR_H5, "r") as hf:
        pids = sorted(hf.keys())
        print(f"\n  ClinVar proteins in H5: {len(pids)}")
        for pid in pids:
            grp = hf[pid]
            if "wt" not in grp or "variant_meta" not in grp.attrs:
                continue
            try:
                meta_list = json.loads(grp.attrs["variant_meta"])
            except (json.JSONDecodeError, KeyError):
                continue
            valid = [
                m for m in meta_list
                if m.get("label") is not None and f"v_{m['idx']}" in grp
            ]
            if not valid:
                continue
            cv_wt_embs[pid] = grp["wt"][:].astype(np.float32)
            cv_meta[pid] = valid

    if not cv_wt_embs:
        return {"skipped": True, "reason": "no valid ClinVar proteins"}

    n_total = sum(len(m) for m in cv_meta.values())
    print(f"  Loaded {len(cv_wt_embs)} WT embeddings, {n_total} labelled variants total")

    # Extend fit_corpus with ClinVar WTs for centering
    full_corpus = {**fit_corpus, **cv_wt_embs}

    codec_auc: dict[str, dict] = {}
    for codec_name in codec_names:
        spec = ALL_CODEC_SPECS[codec_name]
        print(f"\n  [{codec_name}]", flush=True)
        t0 = time.time()
        try:
            codec_kwargs = {
                "d_out": spec.d_out,
                "quantization": spec.quantization,
                "abtt_k": spec.abtt_k,
            }
            if spec.quantization == "pq":
                codec_kwargs["pq_m"] = spec.pq_m
            codec = OneEmbeddingCodec(**codec_kwargs)
            codec.fit(full_corpus)

            all_scores: list[float] = []
            all_labels: list[int] = []

            # Re-open H5 once for streaming variant reads.
            with h5py.File(OUT_CLINVAR_H5, "r") as hf:
                for pid, meta in cv_meta.items():
                    grp = hf[pid]
                    wt_raw = cv_wt_embs[pid]
                    wt_dec = codec.decode_per_residue(codec.encode(wt_raw))
                    for m in meta:
                        mut_raw = grp[f"v_{m['idx']}"][:].astype(np.float32)
                        mut_dec = codec.decode_per_residue(codec.encode(mut_raw))
                        score = score_clinvar_zeroshot(wt_dec, mut_dec, int(m["mut_pos"]))
                        all_scores.append(score)
                        all_labels.append(int(m["label"]))
                        # mut_raw/mut_dec drop on next iter

            scores_arr = np.array(all_scores, dtype=np.float32)
            labels_arr = np.array(all_labels, dtype=np.int32)
            auc = clinvar_auc(scores_arr, labels_arr)
            codec_auc[codec_name] = {
                "auc": float(auc),
                "n_variants": len(all_scores),
                "n_proteins": len(cv_wt_embs),
            }
            print(f"    AUC={auc:.3f}  n={len(all_scores)}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            import traceback; traceback.print_exc()
            codec_auc[codec_name] = {"error": str(exc)}

    # Summary
    print(f"\n{'─'*50}")
    print(f"{'Codec':<16} {'AUC':>8} {'N variants':>12}")
    print(f"{'─'*50}")
    for cname, s in codec_auc.items():
        if "error" in s:
            print(f"  {cname:<16} ERROR  {s['error'][:30]}")
        else:
            print(f"  {cname:<16} {s['auc']:>8.3f} {s['n_variants']:>12}")

    return {"codecs": codec_auc}


# ── Section 3: RNS ride-along ──────────────────────────────────────────────

def run_rns(
    wt_embs: dict[str, np.ndarray],
    wt_seqs: dict[str, str],
    rho_per_codec: dict[str, dict[str, float]],
    codec_names: list[str],
    fit_corpus: dict[str, np.ndarray],
    n_boot: int = BOOTSTRAP_N,
) -> dict:
    """Compute RNS per assay; correlate RNS with VEP ρ.

    RNS is computed on the raw WT embeddings. The correlation is measured
    for both the raw ('lossless') VEP ρ and the binary_896 VEP ρ.
    """
    print(f"\n{'='*70}")
    print("RNS RIDE-ALONG")
    print(f"{'='*70}")

    if len(wt_embs) < 3:
        print(f"  [SKIP] too few assays ({len(wt_embs)}) for meaningful correlation", flush=True)
        return {"skipped": True, "reason": f"only {len(wt_embs)} assays"}

    print(f"\n  Computing RNS for {len(wt_embs)} WT proteins ...", flush=True)
    t0 = time.time()
    try:
        rns_scores = compute_rns_for_assays(
            wt_embs=wt_embs,
            wt_sequences=wt_seqs,
            n_shuffles=5,
            k=min(100, len(wt_embs) - 1),  # k < n for small sets
            seed=42,
        )
    except Exception as exc:
        print(f"  [SKIP RNS] {exc}", flush=True)
        return {"skipped": True, "reason": str(exc)}
    print(f"  RNS computed in {time.time()-t0:.1f}s")

    assay_ids = sorted(rns_scores.keys())
    rns_vals = np.array([rns_scores[a] for a in assay_ids], dtype=np.float64)

    print(f"\n  RNS per assay:")
    for a in assay_ids:
        print(f"    {a}: RNS={rns_scores[a]:.3f}")

    # Compute per-codec RNS
    codec_rns_results: dict[str, np.ndarray] = {}
    for codec_name in codec_names:
        if codec_name == "lossless":
            codec_rns_results[codec_name] = rns_vals
            continue
        spec = ALL_CODEC_SPECS[codec_name]
        print(f"\n  RNS for codec {codec_name} ...", flush=True)
        try:
            codec = OneEmbeddingCodec(
                d_out=spec.d_out,
                quantization=spec.quantization,
                pq_m=spec.pq_m if spec.pq_m else None,
                abtt_k=spec.abtt_k,
            )
            codec.fit(fit_corpus)
            dec_embs = {
                aid: codec.decode_per_residue(codec.encode(wt_embs[aid]))
                for aid in assay_ids
            }
            rns_codec = compute_rns_for_assays(
                wt_embs=dec_embs,
                wt_sequences={a: wt_seqs[a] for a in assay_ids},
                n_shuffles=5,
                k=min(100, len(dec_embs) - 1),
                seed=42,
            )
            codec_rns_results[codec_name] = np.array(
                [rns_codec.get(a, float("nan")) for a in assay_ids]
            )
        except Exception as exc:
            print(f"    [SKIP] {exc}", flush=True)

    # Correlate RNS (raw) with VEP ρ for each codec
    rns_vep_correlations: dict[str, dict] = {}
    focus_codecs = [c for c in ["lossless", "binary_896"] if c in codec_names]
    for codec_name in focus_codecs:
        vep_rho_vals = np.array(
            [rho_per_codec.get(codec_name, {}).get(a, float("nan")) for a in assay_ids]
        )
        # Drop NaN pairs
        valid = ~(np.isnan(rns_vals) | np.isnan(vep_rho_vals))
        if valid.sum() < 3:
            rns_vep_correlations[codec_name] = {"skipped": True, "reason": "too few valid pairs"}
            continue
        try:
            corr = bootstrap_ci_pearson(
                rns_vals[valid], vep_rho_vals[valid], n_boot=n_boot, seed=42
            )
            rns_vep_correlations[codec_name] = corr
            print(f"\n  RNS ↔ VEP-ρ ({codec_name}): r={corr['pearson_r']:.3f} "
                  f"[{corr['ci_low']:.3f}, {corr['ci_high']:.3f}]  n={corr['n']}")
        except Exception as exc:
            rns_vep_correlations[codec_name] = {"error": str(exc)}

    return {
        "rns_per_assay": {a: float(rns_scores[a]) for a in assay_ids},
        "rns_vep_correlation": rns_vep_correlations,
        "codec_rns": {
            c: {a: float(v) for a, v in zip(assay_ids, vals)}
            for c, vals in codec_rns_results.items()
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 56: VEP codec mega-sweep")
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=list(ALL_CODEC_SPECS.keys()),
        default=None,
        help="Codec names to evaluate (default: all 5 tiers).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Reduced run: lossless + binary_896_abtt3 + int2_896.",
    )
    parser.add_argument("--skip-clinvar", action="store_true", help="Skip ClinVar AUC section.")
    parser.add_argument("--skip-rns", action="store_true", help="Skip RNS ride-along section.")
    args = parser.parse_args()

    # Resolve codec list
    if args.smoke_test:
        codec_names = SMOKE_CODEC_NAMES
        print("=== SMOKE TEST: lossless + binary_896_abtt3 + int2_896 ===")
    elif args.codecs:
        codec_names = args.codecs
    else:
        codec_names = DEFAULT_CODEC_NAMES

    print(f"\nCodecs: {codec_names}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    output: dict = {
        "experiment": "exp56_vep_codec_megasweep",
        "codecs_evaluated": codec_names,
        "smoke_test": args.smoke_test,
    }

    # ── Section 1: DMS ──
    dms_results, wt_embs, wt_seqs = run_dms_retention(
        codec_names=codec_names,
        seeds=SEEDS,
        n_boot=BOOTSTRAP_N,
    )
    output["dms_retention"] = dms_results

    # Build rho_per_codec from dms_results for RNS section
    rho_per_codec: dict[str, dict[str, float]] = {}
    for cname in codec_names:
        per_assay = dms_results["codecs"].get(cname, {}).get("per_assay_rho", {})
        rho_per_codec[cname] = per_assay

    fit_corpus = {aid: wt_embs[aid] for aid in wt_embs}

    # ── Section 2: ClinVar ──
    if args.skip_clinvar:
        output["clinvar_auc"] = {"skipped": True, "reason": "--skip-clinvar flag"}
    else:
        output["clinvar_auc"] = run_clinvar(
            codec_names=codec_names,
            fit_corpus=fit_corpus,
        )

    # ── Section 3: RNS ──
    if args.skip_rns:
        output["rns"] = {"skipped": True, "reason": "--skip-rns flag"}
    else:
        output["rns"] = run_rns(
            wt_embs=wt_embs,
            wt_seqs=wt_seqs,
            rho_per_codec=rho_per_codec,
            codec_names=codec_names,
            fit_corpus=fit_corpus,
            n_boot=BOOTSTRAP_N,
        )

    output["total_time_s"] = float(time.time() - t_start)

    # ── Save JSON ──
    out_path = RESULTS_DIR / "exp56_vep_codec_megasweep.json"

    def _json_default(x):
        if hasattr(x, "item"):
            return float(x)
        return str(x)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"\nResults saved: {out_path}")
    print(f"Total time: {output['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
