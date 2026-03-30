#!/usr/bin/env python3
"""Phase D — Progressive Codec benchmark and packaging.

Benchmarks all V2 codec modes end-to-end: fit codebook, encode, decode,
evaluate retrieval + per-residue, measure storage.

Steps:
  D1: Fit codebook on training set, benchmark all modes
  D2: Roundtrip test (encode → save → load → decode → evaluate)
"""

import json
import random
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.codec_v2 import OneEmbeddingCodec

# V2 mode configs (512d for historical reproducibility with Exp 34 tiers)
V2_CONFIGS = {
    "full":     {"d_out": 512, "quantization": "int4",   "pq_m": None, "desc": "int4 per-residue (V1 compatible)", "type": "int4"},
    "balanced": {"d_out": 512, "quantization": "pq",     "pq_m": 128,  "desc": "PQ M=128",                         "type": "pq"},
    "compact":  {"d_out": 512, "quantization": "pq",     "pq_m": 64,   "desc": "PQ M=64",                          "type": "pq"},
    "micro":    {"d_out": 512, "quantization": "pq",     "pq_m": 32,   "desc": "PQ M=32",                          "type": "pq"},
    "binary":   {"d_out": 512, "quantization": "binary", "pq_m": None, "desc": "1-bit sign quantization",          "type": "binary"},
}

from src.one_embedding.transforms import dct_summary
from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.quantization import (
    quantize_int4, dequantize_int4,
    quantize_binary, dequantize_binary,
    pq_encode, pq_decode,
)
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe, evaluate_ss8_probe,
    evaluate_disorder_probe, evaluate_tm_probe,
    load_cb513_csv, load_chezod_seth, load_tmbed_annotated,
)
from src.utils.h5_store import load_residue_embeddings
from src.extraction.data_loader import load_metadata_csv, filter_by_family_size

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "progressive_codec_results.json"
CODEBOOK_DIR = DATA_DIR / "codebooks"
SPLIT_PATH = DATA_DIR / "splits" / "esm2_650m_5k_split.json"

# ── Helpers ───────────────────────────────────────────────────────────────

def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"steps_done": []}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def load_split():
    with open(SPLIT_PATH) as f:
        s = json.load(f)
    return s["train_ids"], s["test_ids"]


def load_metadata():
    meta = load_metadata_csv(DATA_DIR / "proteins" / "metadata_5k.csv")
    meta, _ = filter_by_family_size(meta, min_members=3)
    return meta


# ── Step D1: Full benchmark of all modes ──────────────────────────────────

def step_D1(results):
    print("\n" + "=" * 60)
    print("STEP D1: Benchmark all V2 codec modes")
    print("=" * 60)

    train_ids, test_ids = load_split()
    metadata = load_metadata()

    # Load raw embeddings
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    train_embs = {k: v for k, v in embeddings.items() if k in set(train_ids)}
    print(f"  Loaded {len(embeddings)} proteins ({len(train_embs)} train)")

    # Load CB513 for SS3/SS8
    cb513_raw = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    )
    cb513_path = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
    _, ss3_labels, ss8_labels, _ = load_cb513_csv(cb513_path)
    cb513_avail = sorted(set(cb513_raw.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]

    # Load disorder data
    seth_dir = DATA_DIR / "per_residue_benchmarks"
    seth_fasta = seth_dir / "SETH" / "CheZOD1174_training_set_sequences.fasta"
    disorder_data = None
    if seth_fasta.exists():
        _, disorder_scores, train_ids_d, test_ids_d = load_chezod_seth(seth_dir)
        disorder_embs = {}
        with __import__("h5py").File(
            str(DATA_DIR / "residue_embeddings" / "prot_t5_xl_validation.h5"), "r"
        ) as f:
            for k in f.keys():
                if k.startswith("chezod_"):
                    disorder_embs[k[7:]] = np.array(f[k], dtype=np.float32)
        disorder_data = (disorder_embs, disorder_scores, train_ids_d, test_ids_d)
        print(f"  Disorder data: {len(disorder_embs)} proteins")

    # Load TM data
    tmbed_path = DATA_DIR / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"
    tm_data = None
    if tmbed_path.exists():
        tm_seq, tm_labels = load_tmbed_annotated(tmbed_path)
        tm_embs = {}
        with __import__("h5py").File(
            str(DATA_DIR / "residue_embeddings" / "prot_t5_xl_validation.h5"), "r"
        ) as f:
            for k in f.keys():
                if k.startswith("tmbed_"):
                    tm_embs[k[6:]] = np.array(f[k], dtype=np.float32)
        tm_avail = sorted(set(tm_embs.keys()) & set(tm_labels.keys()))
        rng_tm = random.Random(42)
        rng_tm.shuffle(tm_avail)
        n_tm = int(len(tm_avail) * 0.8)
        tm_data = (tm_embs, tm_labels, tm_avail[:n_tm], tm_avail[n_tm:])
        print(f"  TMbed data: {len(tm_embs)} proteins")

    CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    d1_results = {}
    modes_to_test = ["full", "balanced", "compact", "micro", "binary"]

    for mode in modes_to_test:
        print(f"\n  ═══ Mode: {mode} ({V2_CONFIGS[mode]['desc']}) ═══")
        t0 = time.time()

        # Create and fit codec
        cfg = V2_CONFIGS[mode]
        codec = OneEmbeddingCodec(d_out=cfg["d_out"], quantization=cfg["quantization"], pq_m=cfg["pq_m"])
        print(f"    Fitting codebook on {len(train_embs)} train proteins...")
        codec.fit(train_embs)

        codebook_path = CODEBOOK_DIR / f"codebook_{mode}.h5"
        codec.save_codebook(codebook_path)
        print(f"    Codebook saved: {codebook_path}")

        # Reload codec from saved codebook (tests persistence)
        codec = OneEmbeddingCodec(d_out=cfg["d_out"], quantization=cfg["quantization"], pq_m=cfg["pq_m"], codebook_path=str(codebook_path))

        # Encode all 5K proteins
        print(f"    Encoding {len(embeddings)} proteins...")
        encoded_all = {}
        for pid, m in embeddings.items():
            encoded_all[pid] = codec.encode(m)

        # ── Retrieval ──
        vectors = {pid: enc["protein_vec"].astype(np.float32)
                   for pid, enc in encoded_all.items()}
        ret = evaluate_retrieval_from_vectors(
            vectors, metadata, label_key="family",
            query_ids=test_ids, database_ids=test_ids,
        )
        print(f"    Retrieval: Ret@1={ret['precision@1']:.3f} MRR={ret['mrr']:.3f}")

        # ── SS3/SS8 on CB513 ──
        cb513_decoded = {}
        for pid, m in cb513_raw.items():
            enc = codec.encode(m)
            cb513_decoded[pid] = codec.decode_per_residue(enc)

        ss3 = evaluate_ss3_probe(cb513_decoded, ss3_labels, cb_train, cb_test)
        ss8 = evaluate_ss8_probe(cb513_decoded, ss8_labels, cb_train, cb_test)
        print(f"    SS3 Q3={ss3['q3']:.3f}  SS8 Q8={ss8['q8']:.3f}")

        # ── Disorder ──
        disorder_rho = None
        if disorder_data is not None:
            d_embs, d_scores, d_train, d_test = disorder_data
            d_decoded = {}
            for pid, m in d_embs.items():
                if pid in d_scores:
                    enc = codec.encode(m)
                    d_decoded[pid] = codec.decode_per_residue(enc)
            dis = evaluate_disorder_probe(d_decoded, d_scores, d_train, d_test)
            disorder_rho = dis["spearman_rho"]
            print(f"    Disorder rho={disorder_rho:.3f}")

        # ── TM topology ──
        tm_f1 = None
        if tm_data is not None:
            t_embs, t_labels, t_train, t_test = tm_data
            t_decoded = {}
            for pid, m in t_embs.items():
                if pid in t_labels:
                    enc = codec.encode(m)
                    t_decoded[pid] = codec.decode_per_residue(enc)
            tm = evaluate_tm_probe(t_decoded, t_labels, t_train, t_test)
            tm_f1 = tm["macro_f1"]
            print(f"    TM F1={tm_f1:.3f}")

        # ── Storage ──
        mean_L = int(np.mean([m.shape[0] for m in embeddings.values()]))
        # Measure actual bytes from a sample encode
        sample_enc = encoded_all[list(encoded_all.keys())[0]]
        actual_bytes = sample_enc["protein_vec"].nbytes
        if "per_residue_data" in sample_enc:  # int4
            actual_bytes += sample_enc["per_residue_data"].nbytes
        elif "per_residue_bits" in sample_enc:  # binary
            actual_bytes += sample_enc["per_residue_bits"].nbytes
        elif "pq_codes" in sample_enc:  # pq
            actual_bytes += sample_enc["pq_codes"].nbytes
        # Scale to mean protein length
        sample_L = sample_enc["metadata"]["seq_len"]
        per_protein_bytes = int(actual_bytes * mean_L / max(sample_L, 1))

        elapsed = time.time() - t0

        row = {
            "mode": mode,
            "desc": V2_CONFIGS[mode]["desc"],
            "family_ret1": ret["precision@1"],
            "family_mrr": ret["mrr"],
            "ss3_q3": ss3["q3"],
            "ss8_q8": ss8["q8"],
            "disorder_rho": disorder_rho,
            "tm_f1": tm_f1,
            "per_protein_bytes": per_protein_bytes,
            "per_protein_kb": round(per_protein_bytes / 1024, 1),
            "vs_mean_pool": round(per_protein_bytes / 2048, 1),
            "mean_L": mean_L,
            "elapsed_s": round(elapsed, 1),
        }
        d1_results[mode] = row
        print(f"    Size: {per_protein_bytes/1024:.1f} KB ({per_protein_bytes/2048:.1f}x mean pool)")
        print(f"    [{elapsed:.0f}s]")

        del encoded_all

    # ── Add raw ProtT5 baseline ──
    print(f"\n  ═══ Raw ProtT5 baseline ═══")
    from src.evaluation.per_residue_tasks import evaluate_ss3_probe as _ss3
    plm_pr = {}
    for ds_name, ds_data in [
        ("ss3", (cb513_raw, ss3_labels, cb_train, cb_test)),
    ]:
        pass  # Already have baseline from plm benchmark suite

    results["D1"] = d1_results
    results["steps_done"].append("D1")
    save_results(results)
    print("\n  D1 complete!")
    return results


# ── Step D2: Roundtrip save/load test ─────────────────────────────────────

def step_D2(results):
    print("\n" + "=" * 60)
    print("STEP D2: Roundtrip save → load → verify")
    print("=" * 60)

    train_ids, _ = load_split()
    embeddings = load_residue_embeddings(
        DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    )
    train_embs = {k: v for k, v in embeddings.items() if k in set(train_ids)}

    # Pick one test protein
    test_pid = [k for k in embeddings if k not in set(train_ids)][0]
    test_raw = embeddings[test_pid]
    print(f"  Test protein: {test_pid} (L={test_raw.shape[0]})")

    d2_results = {}

    for mode in ["compact", "binary", "full"]:
        print(f"\n  --- Roundtrip: {mode} ---")

        cfg_d2 = V2_CONFIGS[mode]
        codec = OneEmbeddingCodec(d_out=cfg_d2["d_out"], quantization=cfg_d2["quantization"], pq_m=cfg_d2["pq_m"])
        codec.fit(train_embs)
        codebook_path = CODEBOOK_DIR / f"codebook_{mode}.h5"
        codec.save_codebook(codebook_path)

        # Encode
        encoded = codec.encode(test_raw)
        per_residue_direct = codec.decode_per_residue(encoded)

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name

        codec.save(encoded, tmp_path, protein_id=test_pid)

        cb_path = str(codebook_path) if V2_CONFIGS[mode]["type"] == "pq" else None
        loaded = OneEmbeddingCodec.load(tmp_path, codebook_path=cb_path)

        # Verify
        vec_match = np.allclose(
            encoded["protein_vec"], loaded["protein_vec"], atol=1e-3
        )
        pr_match = np.allclose(per_residue_direct, loaded["per_residue"], atol=1e-4)

        Path(tmp_path).unlink()

        status = "PASS" if (vec_match and pr_match) else "FAIL"
        print(f"    protein_vec match: {vec_match}")
        print(f"    per_residue match: {pr_match}")
        print(f"    → {status}")

        d2_results[mode] = {
            "mode": mode,
            "vec_match": vec_match,
            "pr_match": pr_match,
            "status": status,
        }

    results["D2"] = d2_results
    results["steps_done"].append("D2")
    save_results(results)
    print("\n  D2 complete!")
    return results


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 70)
    print("PHASE D: PROGRESSIVE CODEC BENCHMARK")
    print("=" * 70)

    if "D1" not in results:
        print("  No results yet.")
        return

    print(f"{'Mode':<12s} {'Ret@1':>7s} {'SS3':>7s} {'SS8':>7s} "
          f"{'Dis.ρ':>7s} {'TM F1':>7s} {'KB':>7s} {'×mean':>6s}")
    print("-" * 70)

    for mode in ["full", "balanced", "compact", "micro", "binary"]:
        if mode not in results["D1"]:
            continue
        r = results["D1"][mode]
        dis = f"{r['disorder_rho']:.3f}" if r.get("disorder_rho") else "  N/A"
        tm = f"{r['tm_f1']:.3f}" if r.get("tm_f1") else "  N/A"
        print(f"  {mode:<10s} {r['family_ret1']:>7.3f} {r['ss3_q3']:>7.3f} "
              f"{r['ss8_q8']:>7.3f} {dis:>7s} {tm:>7s} "
              f"{r['per_protein_kb']:>7.1f} {r['vs_mean_pool']:>5.1f}x")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = load_results()

    step = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--step"):
                step = sys.argv[sys.argv.index(arg) + 1]

    if step is None or step == "D1":
        if "D1" not in results.get("steps_done", []):
            results = step_D1(results)
        else:
            print("D1 already done, skipping")

    if step is None or step == "D2":
        if "D2" not in results.get("steps_done", []):
            results = step_D2(results)
        else:
            print("D2 already done, skipping")

    print_summary(results)
    print(f"\nResults saved to {RESULTS_PATH}")
