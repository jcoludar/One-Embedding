"""Exp 48c — RNS across the codec configuration space.

Question: which knobs matter for RNS, and how do PQ / fp16 / int4 compare
to binary at the two pools (mean, DCT-K=4)?

Sweeps:
- quantization: lossless 1024d (skip RP), fp16 896d, int4 896d, binary 896d,
                PQ M=128/192/224 896d
- d_out: 512, 768, 896, 1024 (binary; 1024 skips RP)
- abtt_k: 3 (the pre-Exp-45 default, known to nuke disorder)

All conditions use ``exclude_shuffles_of_query=True``. Junkyard from Exp 48a.

Output:
    data/benchmarks/rigorous_v1/exp48c_knob_sweep.json

Usage:
    uv run python experiments/48c_rns_knob_sweep.py
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
OUT_JSON = ROOT / "data" / "benchmarks" / "rigorous_v1" / "exp48c_knob_sweep.json"

K_NEIGHBORS = 100
DCT_K = 4
SEED = 42


# (label, kwargs to OneEmbeddingCodec) — these instantiate cleanly with .fit() on real CB513.
CONFIGS: list[tuple[str, dict]] = [
    ("raw_baseline",            {}),  # special — handled below; no codec
    ("lossless_1024d_fp16",     dict(d_out=1024, quantization=None, abtt_k=0)),
    ("fp16_896d",               dict(d_out=896,  quantization=None, abtt_k=0)),
    ("int4_896d",               dict(d_out=896,  quantization="int4",   abtt_k=0)),
    ("binary_896d_default",     dict(d_out=896,  quantization="binary", abtt_k=0)),
    ("pq_M64_896d",             dict(d_out=896,  quantization="pq", pq_m=64,  abtt_k=0)),
    ("pq_M128_896d",            dict(d_out=896,  quantization="pq", pq_m=128, abtt_k=0)),
    ("pq_M224_896d",            dict(d_out=896,  quantization="pq", pq_m=224, abtt_k=0)),
    ("binary_512d",             dict(d_out=512,  quantization="binary", abtt_k=0)),
    ("binary_768d",             dict(d_out=768,  quantization="binary", abtt_k=0)),
    ("binary_1024d_no_rp",      dict(d_out=1024, quantization="binary", abtt_k=0)),
    ("binary_896d_abtt3",       dict(d_out=896,  quantization="binary", abtt_k=3)),
]


def load_h5_dict(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {pid: f[pid][:].astype(np.float32) for pid in f.keys()}


def pool_mean(matrix: np.ndarray) -> np.ndarray:
    return matrix.mean(axis=0).astype(np.float32)


def pool_dct(matrix: np.ndarray, K: int = DCT_K) -> np.ndarray:
    return dct_summary(matrix, K=K)


def transform(label: str, kwargs: dict, real: dict, junk: dict):
    """Return (real_decoded, junk_decoded) per-residue arrays for this config."""
    if label == "raw_baseline":
        return real, junk
    codec = OneEmbeddingCodec(**kwargs)
    codec.fit(real)
    real_out: dict[str, np.ndarray] = {}
    junk_out: dict[str, np.ndarray] = {}
    for pid, arr in real.items():
        enc = codec.encode(arr)
        real_out[pid] = codec.decode_per_residue(enc).astype(np.float32)
    for pid, arr in junk.items():
        enc = codec.encode(arr)
        junk_out[pid] = codec.decode_per_residue(enc).astype(np.float32)
    return real_out, junk_out


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
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
        "frac_increased": float((deltas > 0).mean()),
    }


def main() -> None:
    print("== Exp 48c — RNS across codec knobs ==")
    t0 = time.time()

    print(f"Loading {REAL_H5.name} ...")
    real = load_h5_dict(REAL_H5)
    print(f"Loading {JUNK_H5.name} ...")
    junk = load_h5_dict(JUNK_H5)
    print(f"  n_real = {len(real)}, n_junk = {len(junk)}")

    rns_table: dict[str, dict[str, dict[str, float]]] = {}
    print(f"\n{'config':<26} {'pool':<5} {'dim':>5}  {'mean RNS':>10}  CI                 wall")
    for label, kwargs in CONFIGS:
        ts = time.time()
        real_dec, junk_dec = transform(label, kwargs, real, junk)
        rns_table[label] = {}
        for pool_name, fn in {"mean": pool_mean, "dct4": pool_dct}.items():
            real_vecs = {pid: fn(arr) for pid, arr in real_dec.items()}
            junk_vecs = {pid: fn(arr) for pid, arr in junk_dec.items()}
            scores = compute_rns(
                query_vectors=real_vecs,
                real_vectors=real_vecs,
                junkyard_vectors=junk_vecs,
                k=K_NEIGHBORS,
                exclude_shuffles_of_query=True,
            )
            rns_table[label][pool_name] = scores
            agg = aggregate(scores)
            d = next(iter(real_vecs.values())).shape[0]
            print(f"  {label:<24} {pool_name:<5} {d:>5}  {agg['mean']:>10.4f}  "
                  f"[{agg['ci95_lo']:.4f}, {agg['ci95_hi']:.4f}]  "
                  f"{time.time() - ts:.1f}s")

    # Paired vs raw_baseline for each config
    print("\nPaired delta vs raw baseline (per-protein):")
    print(f"  {'config':<24} {'pool':<5}  Δ        CI                  frac_up")
    paired: dict[str, dict[str, dict[str, float]]] = {}
    base = rns_table["raw_baseline"]
    for label, _ in CONFIGS:
        if label == "raw_baseline":
            continue
        paired[label] = {}
        for pool_name in ("mean", "dct4"):
            d = paired_delta(base[pool_name], rns_table[label][pool_name])
            sig = "*" if (d["ci95_lo"] > 0 or d["ci95_hi"] < 0) else " "
            print(f"  {label:<24} {pool_name:<5} {d['mean_delta']:+.4f}  "
                  f"[{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}] {sig} "
                  f"{d['frac_increased']:.1%}")
            paired[label][pool_name] = d

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "real_h5": str(REAL_H5.relative_to(ROOT)),
            "junk_h5": str(JUNK_H5.relative_to(ROOT)),
            "k_neighbors": K_NEIGHBORS,
            "dct_k": DCT_K,
            "seed": SEED,
            "exclude_shuffles_of_query": True,
        },
        "configs": [{"label": l, "kwargs": kw} for l, kw in CONFIGS],
        "aggregate": {
            label: {pool: aggregate(scores) for pool, scores in pools.items()}
            for label, pools in rns_table.items()
        },
        "paired_vs_raw": paired,
        "per_protein": {
            label: {pool: {pid: float(v) for pid, v in scores.items()}
                    for pool, scores in pools.items()}
            for label, pools in rns_table.items()
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
