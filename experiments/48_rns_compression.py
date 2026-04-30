"""Exp 48 — Random Neighbor Score (RNS) under OneEmbedding compression.

Question: does the codec change embedding quality as measured by RNS
(Prabakaran & Bromberg, Nat Methods 2026)?

Design — 2x2 grid:
                       mean pool          DCT-K=4
    raw ProtT5 1024d   C1 (1024,)         C2 (4096,)
    OE 896d binary     C3 (896,)          C4 (3584,)

C4 is what the codec actually ships as ``protein_vec``. The headline
comparison is C2 vs C4 (does compression move RNS for the codec's
shipping protein vector?).

Inputs:
    data/residue_embeddings/prot_t5_xl_cb513.h5            (511 real)
    data/residue_embeddings/prot_t5_xl_cb513_junkyard.h5   (2555 shuffled)

Output:
    data/benchmarks/rigorous_v1/exp48_rns_compression.json

Usage:
    uv run python experiments/48_rns_compression.py
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
OUT_JSON = ROOT / "data" / "benchmarks" / "rigorous_v1" / "exp48_rns_compression.json"

K_NEIGHBORS = 100
DCT_K = 4
SEED = 42


def load_h5_dict(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {pid: f[pid][:].astype(np.float32) for pid in f.keys()}


def pool_mean(matrix: np.ndarray) -> np.ndarray:
    """Mean pool along the length axis. Matches DCT K=1 up to a normalization."""
    return matrix.mean(axis=0).astype(np.float32)


def pool_dct(matrix: np.ndarray, K: int = DCT_K) -> np.ndarray:
    """DCT-K pool — the codec's shipping protein vector pooling."""
    return dct_summary(matrix, K=K)


def build_protein_vectors(
    per_residue: dict[str, np.ndarray],
    pool_fn,
) -> dict[str, np.ndarray]:
    return {pid: pool_fn(arr) for pid, arr in per_residue.items()}


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
    """RNS(b) - RNS(a) per protein, aggregated."""
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
    print("== Exp 48 — RNS under OneEmbedding compression ==")
    t0 = time.time()

    print(f"Loading {REAL_H5.name} ...")
    real = load_h5_dict(REAL_H5)
    print(f"  n_real = {len(real)}, example shape: {next(iter(real.values())).shape}")

    print(f"Loading {JUNK_H5.name} ...")
    junk = load_h5_dict(JUNK_H5)
    print(f"  n_junk = {len(junk)}, example shape: {next(iter(junk.values())).shape}")

    # Fit codec on real CB513 (centering stats only — binary mode has no codebook)
    print("Fitting OneEmbeddingCodec (binary 896d, default) ...")
    codec = OneEmbeddingCodec()
    codec.fit(real)

    # Decode OneEmbedding back to (L, 896) float for both real and junk
    print("Encoding + decoding through codec ...")
    oe_real: dict[str, np.ndarray] = {}
    for pid, arr in real.items():
        enc = codec.encode(arr)
        oe_real[pid] = codec.decode_per_residue(enc).astype(np.float32)

    oe_junk: dict[str, np.ndarray] = {}
    for pid, arr in junk.items():
        enc = codec.encode(arr)
        oe_junk[pid] = codec.decode_per_residue(enc).astype(np.float32)

    print(f"  OE shapes: real {next(iter(oe_real.values())).shape}, "
          f"junk {next(iter(oe_junk.values())).shape}")

    # Build protein vectors for the 4 conditions
    print("Pooling 4 conditions ...")
    conditions = {
        "C1_raw_mean":    (real,    pool_mean),
        "C2_raw_dct4":    (real,    pool_dct),
        "C3_oe_mean":     (oe_real, pool_mean),
        "C4_oe_dct4":     (oe_real, pool_dct),
    }
    junk_conditions = {
        "C1_raw_mean":    (junk,    pool_mean),
        "C2_raw_dct4":    (junk,    pool_dct),
        "C3_oe_mean":     (oe_junk, pool_mean),
        "C4_oe_dct4":     (oe_junk, pool_dct),
    }

    rns_per_condition: dict[str, dict[str, float]] = {}
    for name, (per_res, fn) in conditions.items():
        real_vecs = build_protein_vectors(per_res, fn)
        junk_vecs = build_protein_vectors(junk_conditions[name][0], fn)
        # query = real proteins
        scores = compute_rns(
            query_vectors=real_vecs,
            real_vectors=real_vecs,
            junkyard_vectors=junk_vecs,
            k=K_NEIGHBORS,
        )
        rns_per_condition[name] = scores
        agg = aggregate(scores)
        print(f"  {name}: dim={next(iter(real_vecs.values())).shape[0]:>4}  "
              f"mean RNS={agg['mean']:.4f}  CI=[{agg['ci95_lo']:.4f}, {agg['ci95_hi']:.4f}]  "
              f"median={agg['median']:.4f}")

    # Aggregate stats + paired deltas (key comparisons)
    print("Computing paired deltas ...")
    paired = {
        "raw_pooling__C2_minus_C1":   paired_delta(rns_per_condition["C1_raw_mean"], rns_per_condition["C2_raw_dct4"]),
        "oe_pooling__C4_minus_C3":    paired_delta(rns_per_condition["C3_oe_mean"], rns_per_condition["C4_oe_dct4"]),
        "compression_mean__C3_minus_C1": paired_delta(rns_per_condition["C1_raw_mean"], rns_per_condition["C3_oe_mean"]),
        "compression_dct4__C4_minus_C2": paired_delta(rns_per_condition["C2_raw_dct4"], rns_per_condition["C4_oe_dct4"]),
    }
    for name, d in paired.items():
        sig = "*" if (d["ci95_lo"] > 0 or d["ci95_hi"] < 0) else " "
        print(f"  {name}: Δ={d['mean_delta']:+.4f}  "
              f"CI=[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}] {sig}  "
              f"frac_up={d['frac_increased']:.2%}")

    # Save full output
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "real_h5": str(REAL_H5.relative_to(ROOT)),
            "junk_h5": str(JUNK_H5.relative_to(ROOT)),
            "k_neighbors": K_NEIGHBORS,
            "dct_k": DCT_K,
            "seed": SEED,
            "codec": "OneEmbeddingCodec(d_out=896, quantization='binary', abtt_k=0)",
        },
        "aggregate": {name: aggregate(scores) for name, scores in rns_per_condition.items()},
        "paired": paired,
        "per_protein": {
            name: {pid: float(v) for pid, v in scores.items()}
            for name, scores in rns_per_condition.items()
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
