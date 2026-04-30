"""Exp 48b — Localize the RNS shift across the OneEmbedding pipeline.

Exp 48 found a +0.14 RNS shift at DCT-K=4 between raw ProtT5 and the OE
codec, robust to filtering same-source shuffles. This experiment splits
the pipeline into stages and computes RNS at each, isolating which step
contributes the delta.

Pipeline stages (cumulative):
    S0: raw 1024d                                      (baseline)
    S1: centered 1024d                                 (+ centering)
    S2: centered + RP896 fp32                          (+ random projection)
    S3: centered + RP896 fp16                          (+ fp16 cast)
    S4: centered + RP896 binary (decoded back to fp32) (+ binary quantization) ← OE default

Each stage pooled by mean and by DCT-K=4 → 10 conditions total.

RNS uses ``exclude_shuffles_of_query=True`` (own-shuffles filtered).

Output:
    data/benchmarks/rigorous_v1/exp48b_rns_ablation.json

Usage:
    uv run python experiments/48b_rns_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.one_embedding.rns import compute_rns
from src.one_embedding.transforms import dct_summary


REAL_H5 = ROOT / "data" / "residue_embeddings" / "prot_t5_xl_cb513.h5"
JUNK_H5 = ROOT / "data" / "residue_embeddings" / "prot_t5_xl_cb513_junkyard.h5"
OUT_JSON = ROOT / "data" / "benchmarks" / "rigorous_v1" / "exp48b_rns_ablation.json"

K_NEIGHBORS = 100
DCT_K = 4
SEED = 42


def load_h5_dict(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {pid: f[pid][:].astype(np.float32) for pid in f.keys()}


def pool_mean(matrix: np.ndarray) -> np.ndarray:
    return matrix.mean(axis=0).astype(np.float32)


def pool_dct(matrix: np.ndarray, K: int = DCT_K) -> np.ndarray:
    return dct_summary(matrix, K=K)


def stage_transform(
    name: str,
    per_residue: dict[str, np.ndarray],
    codec: OneEmbeddingCodec,
) -> dict[str, np.ndarray]:
    """Return per-residue arrays at the named pipeline stage."""
    mean_vec = codec._corpus_stats["mean_vec"]
    R = codec._get_projection_matrix(d_in=1024)  # raw ProtT5 dim

    out: dict[str, np.ndarray] = {}
    for pid, raw in per_residue.items():
        x = raw.astype(np.float32)
        if name == "S0_raw":
            out[pid] = x
        elif name == "S1_centered":
            out[pid] = x - mean_vec
        elif name == "S2_rp_fp32":
            out[pid] = (x - mean_vec) @ R
        elif name == "S3_rp_fp16":
            out[pid] = ((x - mean_vec) @ R).astype(np.float16).astype(np.float32)
        elif name == "S4_oe_binary":
            enc = codec.encode(raw)
            out[pid] = codec.decode_per_residue(enc).astype(np.float32)
        else:
            raise ValueError(name)
    return out


def aggregate(scores: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(scores.values()), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    boots = np.array([
        arr[rng.integers(0, len(arr), len(arr))].mean()
        for _ in range(10_000)
    ])
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
    }


def paired_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    common = sorted(set(a) & set(b))
    deltas = np.array([b[pid] - a[pid] for pid in common], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    boots = np.array([
        deltas[rng.integers(0, len(deltas), len(deltas))].mean()
        for _ in range(10_000)
    ])
    return {
        "n": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
        "frac_increased": float((deltas > 0).mean()),
    }


def main() -> None:
    print("== Exp 48b — RNS pipeline ablation ==")
    t0 = time.time()

    print(f"Loading {REAL_H5.name} ...")
    real = load_h5_dict(REAL_H5)
    print(f"  n_real = {len(real)}")

    print(f"Loading {JUNK_H5.name} ...")
    junk = load_h5_dict(JUNK_H5)
    print(f"  n_junk = {len(junk)}")

    codec = OneEmbeddingCodec()
    codec.fit(real)

    stages = ["S0_raw", "S1_centered", "S2_rp_fp32", "S3_rp_fp16", "S4_oe_binary"]
    pools = {"mean": pool_mean, "dct4": pool_dct}

    rns_table: dict[str, dict[str, dict[str, float]]] = {}
    print("\nStage × pool → mean RNS [95% CI]")
    print(f"  {'stage':<14} {'pool':<5} {'dim':>5}  {'mean RNS':>10}  CI")

    for stage in stages:
        real_stage = stage_transform(stage, real, codec)
        junk_stage = stage_transform(stage, junk, codec)
        rns_table[stage] = {}
        for pool_name, fn in pools.items():
            real_vecs = {pid: fn(arr) for pid, arr in real_stage.items()}
            junk_vecs = {pid: fn(arr) for pid, arr in junk_stage.items()}
            scores = compute_rns(
                query_vectors=real_vecs,
                real_vectors=real_vecs,
                junkyard_vectors=junk_vecs,
                k=K_NEIGHBORS,
                exclude_shuffles_of_query=True,
            )
            rns_table[stage][pool_name] = scores
            agg = aggregate(scores)
            d = next(iter(real_vecs.values())).shape[0]
            print(f"  {stage:<14} {pool_name:<5} {d:>5}  {agg['mean']:>10.4f}  "
                  f"[{agg['ci95_lo']:.4f}, {agg['ci95_hi']:.4f}]")

    # Stage-to-stage paired deltas (incremental cost of each step)
    print("\nIncremental stage deltas (paired, per-protein):")
    print(f"  {'transition':<35} {'pool':<5}  Δ           CI                  frac_up")
    paired: dict[str, dict[str, dict[str, float]]] = {}
    for pool_name in pools:
        paired[pool_name] = {}
        for prev, curr in zip(stages[:-1], stages[1:]):
            d = paired_delta(rns_table[prev][pool_name], rns_table[curr][pool_name])
            label = f"{prev} → {curr}"
            sig = "*" if (d["ci95_lo"] > 0 or d["ci95_hi"] < 0) else " "
            print(f"  {label:<35} {pool_name:<5} {d['mean_delta']:+.4f}  "
                  f"[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}] {sig} "
                  f"{d['frac_increased']:.1%}")
            paired[pool_name][label] = d

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "real_h5": str(REAL_H5.relative_to(ROOT)),
            "junk_h5": str(JUNK_H5.relative_to(ROOT)),
            "k_neighbors": K_NEIGHBORS,
            "dct_k": DCT_K,
            "seed": SEED,
            "exclude_shuffles_of_query": True,
            "codec": "OneEmbeddingCodec(d_out=896, quantization='binary', abtt_k=0)",
        },
        "aggregate": {
            stage: {pool: aggregate(scores) for pool, scores in pools_dict.items()}
            for stage, pools_dict in rns_table.items()
        },
        "incremental_deltas": paired,
        "per_protein": {
            stage: {pool: {pid: float(v) for pid, v in scores.items()}
                    for pool, scores in pools_dict.items()}
            for stage, pools_dict in rns_table.items()
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
