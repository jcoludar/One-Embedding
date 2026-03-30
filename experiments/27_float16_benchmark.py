"""Experiment 27: Float16 vs Float32 codec benchmark.

Head-to-head comparison of the OneEmbeddingCodec at float16 (default, 25% of raw)
vs float32 (50% of raw) on real ProtT5-XL embeddings.

Benchmarks:
  - Family retrieval Ret@1 on SCOPe 5K test set (n=850)
  - Per-residue SS3 probe on CB513
  - Per-residue SS8 probe on CB513
  - Max absolute error between float16 and float32 representations

Output:
  - data/benchmarks/float16_benchmark_results.json
  - docs/figures/pub_float16_benchmark.png
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe,
    evaluate_ss8_probe,
    load_cb513_csv,
)
from src.extraction.data_loader import filter_by_family_size, load_metadata_csv
from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.one_embedding.transforms import dct_summary
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.utils.h5_store import load_residue_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
OUT_DIR = DATA_DIR / "benchmarks"
FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"


def load_split() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)


def load_metadata() -> list[dict]:
    meta_path = DATA_DIR / "proteins" / "metadata_5k.csv"
    metadata = load_metadata_csv(meta_path)
    metadata, _ = filter_by_family_size(metadata, min_members=3)
    return metadata


def load_plm_embeddings(plm_stem: str, dataset: str = "medium5k") -> dict[str, np.ndarray]:
    h5_path = DATA_DIR / "residue_embeddings" / f"{plm_stem}_{dataset}.h5"
    if not h5_path.exists():
        print(f"  WARNING: {h5_path} not found")
        return {}
    embeddings = load_residue_embeddings(h5_path)
    # Remap pipe-separated IDs if needed
    needs_remap = any("|" in k for k in list(embeddings.keys())[:5])
    if needs_remap:
        remapped = {}
        for k, v in embeddings.items():
            remapped[k.split("|")[0]] = v
        return remapped
    return embeddings


def normal_ci(p: float, n: int) -> float:
    """95% CI half-width for a proportion (normal approximation)."""
    return 1.96 * np.sqrt(p * (1 - p) / n)


def encode_per_residue(raw: np.ndarray, codec: OneEmbeddingCodec, dtype=np.float16) -> np.ndarray:
    """Apply D-compression only (random projection), return (L, d_out) in target dtype.

    Note: the unified OneEmbeddingCodec always stores as fp16 (quantization=None).
    The dtype parameter is kept for the float32 comparison path in this historical script.
    """
    R = codec._get_projection_matrix(raw.shape[1])
    return (raw @ R).astype(dtype)


def encode_protein_vec(per_residue_f32: np.ndarray, codec: OneEmbeddingCodec, dtype=np.float16) -> np.ndarray:
    """DCT pooling on float32 per-residue, then cast to target dtype."""
    protein_vec = dct_summary(per_residue_f32, K=codec.dct_k)
    return protein_vec.astype(dtype)


def main():
    print("=" * 60)
    print("Experiment 27: Float16 vs Float32 Codec Benchmark")
    print("=" * 60)

    results = {}

    # ── Setup ──
    # Unified codec: quantization=None stores as fp16 (only supported dtype).
    # float32 comparison is historical — kept as an explicit numpy cast below.
    codec16 = OneEmbeddingCodec(d_out=512, quantization=None, dct_k=4, seed=42)
    # codec32 is emulated by casting to float32 explicitly; the unified codec
    # no longer has a dtype parameter since fp32 storage is not supported.
    codec32 = codec16  # same projection matrix; float32 path uses dtype=np.float32 below

    metadata = load_metadata()
    split = load_split()
    test_ids = split["test_ids"]

    # ── Step 1: Retrieval benchmark (SCOPe 5K) ──
    print("\n── Step 1: Retrieval Ret@1 (SCOPe 5K test set) ──")
    embeddings = load_plm_embeddings("prot_t5_xl", "medium5k")
    print(f"  Loaded {len(embeddings)} ProtT5-XL embeddings")

    for label, target_dtype in [("float32", np.float32), ("float16", np.float16)]:
        codec = codec16  # same projection matrix for both; dtype differs only in cast
        t0 = time.time()
        vectors = {}
        for pid in test_ids:
            if pid not in embeddings:
                continue
            raw = embeddings[pid].astype(np.float32)
            R = codec._get_projection_matrix(raw.shape[1])
            per_res_f32 = raw @ R  # always compute in float32
            protein_vec = dct_summary(per_res_f32, K=codec.dct_k)
            # Cast to target dtype, then back to float32 for cosine similarity
            vectors[pid] = protein_vec.astype(target_dtype).astype(np.float32)

        ret = evaluate_retrieval_from_vectors(
            vectors, metadata, label_key="family",
            query_ids=test_ids, database_ids=test_ids, metric="cosine",
        )
        elapsed = time.time() - t0
        n_queries = ret["n_queries"]
        ci = normal_ci(ret["precision@1"], n_queries)

        results[f"retrieval_{label}"] = {
            "ret1": ret["precision@1"],
            "mrr": ret["mrr"],
            "n_queries": n_queries,
            "ci_95": ci,
            "time_s": elapsed,
        }
        print(f"  {label}: Ret@1={ret['precision@1']:.4f} ± {ci:.4f} "
              f"(MRR={ret['mrr']:.4f}, n={n_queries}, {elapsed:.1f}s)")

    # ── Quantization error analysis ──
    print("\n── Quantization error analysis ──")
    max_abs_errors = []
    mean_abs_errors = []
    cos_sims = []
    for pid in test_ids[:200]:  # Sample 200 proteins
        if pid not in embeddings:
            continue
        raw = embeddings[pid].astype(np.float32)
        R = codec16._get_projection_matrix(raw.shape[1])
        per_res_f32 = raw @ R
        pv_f32 = dct_summary(per_res_f32, K=4)
        pv_f16 = pv_f32.astype(np.float16).astype(np.float32)

        max_abs_errors.append(np.max(np.abs(pv_f32 - pv_f16)))
        mean_abs_errors.append(np.mean(np.abs(pv_f32 - pv_f16)))
        cos = np.dot(pv_f32, pv_f16) / (np.linalg.norm(pv_f32) * np.linalg.norm(pv_f16) + 1e-10)
        cos_sims.append(cos)

    results["quantization_error"] = {
        "max_abs_error_median": float(np.median(max_abs_errors)),
        "max_abs_error_95pct": float(np.percentile(max_abs_errors, 95)),
        "mean_abs_error_median": float(np.median(mean_abs_errors)),
        "cosine_similarity_mean": float(np.mean(cos_sims)),
        "cosine_similarity_min": float(np.min(cos_sims)),
        "n_proteins": len(max_abs_errors),
    }
    print(f"  Protein vector max |error| median: {np.median(max_abs_errors):.6f}")
    print(f"  Protein vector max |error| 95th pct: {np.percentile(max_abs_errors, 95):.6f}")
    print(f"  Protein vector cosine sim: {np.mean(cos_sims):.6f} (min={np.min(cos_sims):.6f})")

    del embeddings  # Free memory

    # ── Step 2: Per-residue SS3/SS8 probe (CB513) ──
    print("\n── Step 2: Per-residue SS3/SS8 probes (CB513) ──")
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    cb513_embeddings = load_plm_embeddings("prot_t5_xl", "cb513")

    if cb513_path.exists() and cb513_embeddings:
        sequences, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
        print(f"  CB513: {len(ss3_labels)} proteins, {len(cb513_embeddings)} embeddings")

        for label, target_dtype in [("float32", np.float32), ("float16", np.float16)]:
            codec = codec16  # same projection matrix; only cast differs
            avail_ids = [pid for pid in ss3_labels if pid in cb513_embeddings]
            coded_embs = {}
            for pid in avail_ids:
                raw = cb513_embeddings[pid].astype(np.float32)
                R = codec._get_projection_matrix(raw.shape[1])
                per_res = (raw @ R).astype(target_dtype).astype(np.float32)
                coded_embs[pid] = per_res

            rng = random.Random(42)
            rng.shuffle(avail_ids)
            n_train = int(len(avail_ids) * 0.8)
            train_ids = avail_ids[:n_train]
            test_ids_cb = avail_ids[n_train:]

            print(f"  {label} SS3 ({len(train_ids)} train, {len(test_ids_cb)} test)...")
            ss3 = evaluate_ss3_probe(coded_embs, ss3_labels, train_ids, test_ids_cb)
            print(f"    SS3 Q3 = {ss3.get('q3', 0):.4f}")

            print(f"  {label} SS8 ({len(train_ids)} train, {len(test_ids_cb)} test)...")
            ss8 = evaluate_ss8_probe(coded_embs, ss8_labels, train_ids, test_ids_cb)
            print(f"    SS8 Q8 = {ss8.get('q8', 0):.4f}")

            results[f"ss3_{label}"] = {"q3": ss3.get("q3", 0)}
            results[f"ss8_{label}"] = {"q8": ss8.get("q8", 0)}
    else:
        print("  CB513 not found, skipping SS3/SS8")

    del cb513_embeddings

    # ── Step 3: Storage comparison ──
    print("\n── Step 3: Storage comparison ──")
    mean_L = 175  # Mean sequence length for ProtT5 SCOPe set
    D_raw = 1024

    raw_bytes = mean_L * D_raw * 4  # float32
    f32_per_res = mean_L * 512 * 4
    f32_pvec = 2048 * 4
    f32_total = f32_per_res + f32_pvec

    f16_per_res = mean_L * 512 * 2
    f16_pvec = 2048 * 2
    f16_total = f16_per_res + f16_pvec

    results["storage"] = {
        "raw_bytes": raw_bytes,
        "float32_bytes": f32_total,
        "float16_bytes": f16_total,
        "float32_pct_of_raw": f32_total / raw_bytes * 100,
        "float16_pct_of_raw": f16_total / raw_bytes * 100,
        "float16_pct_of_float32": f16_total / f32_total * 100,
        "mean_L": mean_L,
    }
    print(f"  Raw ProtT5 (L={mean_L}, D=1024): {raw_bytes / 1024:.0f} KB")
    print(f"  Float32 codec (L, 512) + (2048,): {f32_total / 1024:.0f} KB ({f32_total / raw_bytes * 100:.0f}% of raw)")
    print(f"  Float16 codec (L, 512) + (2048,): {f16_total / 1024:.0f} KB ({f16_total / raw_bytes * 100:.0f}% of raw)")

    # ── Save results ──
    out_path = OUT_DIR / "float16_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Step 4: Figure ──
    print("\n── Step 4: Generating figure ──")
    make_figure(results)
    print("\nDone!")


def make_figure(results: dict):
    """Publication-quality figure: float16 vs float32 head-to-head."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Colors
    c32 = "#4A90D9"  # blue for float32
    c16 = "#E8913A"  # orange for float16

    # ── Panel 1: Retrieval Ret@1 ──
    ax = axes[0]
    ret32 = results["retrieval_float32"]
    ret16 = results["retrieval_float16"]
    x = [0, 1]
    vals = [ret32["ret1"], ret16["ret1"]]
    cis = [ret32["ci_95"], ret16["ci_95"]]
    colors = [c32, c16]
    bars = ax.bar(x, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, vals, yerr=cis, fmt="none", capsize=6, color="black", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["float32\n(50% raw)", "float16\n(25% raw)"])
    ax.set_ylabel("Ret@1")
    ax.set_title("Family Retrieval (SCOPe 5K)")
    ax.set_ylim(0.7, max(vals) + 0.06)
    for i, v in enumerate(vals):
        ax.text(i, v + cis[i] + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 2: Per-Residue Probes ──
    ax = axes[1]
    has_ss = "ss3_float32" in results and "ss3_float16" in results
    if has_ss:
        ss3_32 = results["ss3_float32"]["q3"]
        ss3_16 = results["ss3_float16"]["q3"]
        ss8_32 = results["ss8_float32"]["q8"]
        ss8_16 = results["ss8_float16"]["q8"]

        x_pos = np.array([0, 1])
        w = 0.25
        bars1 = ax.bar(x_pos - w/2, [ss3_32, ss8_32], w, color=c32, edgecolor="black",
                       linewidth=0.5, label="float32")
        bars2 = ax.bar(x_pos + w/2, [ss3_16, ss8_16], w, color=c16, edgecolor="black",
                       linewidth=0.5, label="float16")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["SS3 (Q3)", "SS8 (Q8)"])
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Residue Probes (CB513)")
        ax.set_ylim(0.6, max(ss3_32, ss3_16) + 0.06)
        ax.legend(loc="lower right", frameon=False)

        for bar_group in [bars1, bars2]:
            for bar in bar_group:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                       f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "CB513 data\nnot available", transform=ax.transAxes,
               ha="center", va="center", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 3: Storage ──
    ax = axes[2]
    storage = results["storage"]
    raw_kb = storage["raw_bytes"] / 1024
    f32_kb = storage["float32_bytes"] / 1024
    f16_kb = storage["float16_bytes"] / 1024

    x = [0, 1, 2]
    vals = [raw_kb, f32_kb, f16_kb]
    pcts = [100, storage["float32_pct_of_raw"], storage["float16_pct_of_raw"]]
    colors_s = ["#888888", c32, c16]
    bars = ax.bar(x, vals, color=colors_s, width=0.5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Raw ProtT5\n(L,1024)", "Codec fp32\n(L,512)+(2048,)", "Codec fp16\n(L,512)+(2048,)"])
    ax.set_ylabel("KB / protein")
    ax.set_title(f"Storage (mean L={storage['mean_L']})")
    for i, (v, p) in enumerate(zip(vals, pcts)):
        ax.text(i, v + 5, f"{v:.0f} KB\n({p:.0f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Float16 vs Float32 Codec: Quality vs Storage", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = FIG_DIR / "pub_float16_benchmark.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Figure saved to {out_path}")


if __name__ == "__main__":
    main()
