#!/usr/bin/env python3
"""Experiment 36 — One Embedding Toolkit Benchmarks.

Trains and evaluates all per-residue probes on compressed (512d) vs raw (1024d)
embeddings to measure quality retention across the V2 codec.

Benchmarks:
  A. Disorder prediction (CheZOD Z-scores, SETH/UdonPred comparison)
  B. Secondary structure (SS3/SS8 from CB513/TS115)
  C. Conservation scoring (Kibby-style linear probe)
  D. Aligner quality (BAliBASE, if available)
  E. Family classification (SCOPe 5K Ret@1)
  F. Structural similarity (TM-score correlation)

Usage:
  uv run python experiments/36_toolkit_benchmarks.py --task disorder
  uv run python experiments/36_toolkit_benchmarks.py --task all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────

def parse_fasta(path):
    """Parse FASTA file into {id: sequence}."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            seqs[current_id] = "".join(current_seq)
    return seqs


def load_embeddings_and_compress(h5_path, protein_ids=None):
    """Load ProtT5 per-residue embeddings, return both raw (1024d) and compressed (512d).

    Returns:
        raw: {pid: (L, 1024) array}
        compressed: {pid: (L, 512) array}
    """
    import h5py
    from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
    from src.one_embedding.universal_transforms import random_orthogonal_project

    # Load raw embeddings
    raw = {}
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys()) if protein_ids is None else [k for k in protein_ids if k in f]
        for key in keys:
            raw[key] = np.array(f[key], dtype=np.float32)

    print(f"  Loaded {len(raw)} proteins from {h5_path}")

    # Compute ABTT3 stats from corpus
    corpus_h5 = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    if corpus_h5.exists():
        from src.utils.h5_store import load_residue_embeddings
        corpus = load_residue_embeddings(corpus_h5)
        stats = compute_corpus_stats(corpus, n_sample=50_000, n_pcs=5, seed=42)
    else:
        # Use the data itself (less stable but works)
        stats = compute_corpus_stats(raw, n_sample=min(50_000, sum(e.shape[0] for e in raw.values())), n_pcs=5, seed=42)
    top3 = stats["top_pcs"][:3]

    # Compress: ABTT3 + RP512
    compressed = {}
    for pid, emb in raw.items():
        emb_abtt = all_but_the_top(emb, top3)
        emb_rp = random_orthogonal_project(emb_abtt, d_out=512, seed=42)
        compressed[pid] = emb_rp.astype(np.float32)

    print(f"  Compressed: {next(iter(raw.values())).shape[1]}d → {next(iter(compressed.values())).shape[1]}d")
    return raw, compressed


# ── A. Disorder Benchmark ────────────────────────────────────────────────

def benchmark_disorder():
    """Train and evaluate disorder probes on raw vs compressed embeddings."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    print("=" * 60)
    print("A. Disorder Prediction (CheZOD)")
    print("=" * 60)

    DATA_DIR = PROJ_ROOT / "data" / "per_residue_benchmarks" / "SETH"
    EMB_PATH = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_chezod.h5"

    if not EMB_PATH.exists():
        print("  ERROR: CheZOD embeddings not found. Run embedding extraction first.")
        return None

    # Parse training Z-scores
    train_seqs = parse_fasta(DATA_DIR / "CheZOD1174_training_set_sequences.fasta")

    # Parse scores: format is "protein_id:\t score1, score2, ..."
    train_scores = {}
    with open(DATA_DIR / "CheZOD1174_training_set_CheZOD_scores.txt") as f:
        for line in f:
            parts = line.strip().split(":\t")
            if len(parts) == 2:
                pid = parts[0].strip()
                scores = [float(x.strip()) for x in parts[1].split(",") if x.strip()]
                train_scores[pid] = np.array(scores)

    print(f"  Training: {len(train_scores)} proteins with Z-scores")

    # Parse test Z-scores
    test_seqs = parse_fasta(DATA_DIR / "CheZOD117_test_set_sequences.fasta")
    test_scores = {}
    test_dir = DATA_DIR / "CheZOD117_test_scores"
    for f in test_dir.iterdir():
        if f.name.startswith("zscores"):
            pid = f.stem.replace("zscores", "")
            residue_scores = []
            with open(f) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            residue_scores.append(float(parts[2]))
                        except ValueError:
                            pass
            if residue_scores:
                test_scores[pid] = np.array(residue_scores)

    print(f"  Test: {len(test_scores)} proteins with Z-scores")

    # Load and compress embeddings
    all_ids = set(train_scores.keys()) | set(test_scores.keys())
    raw_embs, comp_embs = load_embeddings_and_compress(EMB_PATH, all_ids)

    # Filter to proteins that have both embeddings and scores
    # Match by checking if any key contains the score ID or vice versa
    def match_ids(score_ids, emb_ids):
        """Match score IDs to embedding IDs (may differ in format)."""
        matches = {}
        for sid in score_ids:
            if sid in emb_ids:
                matches[sid] = sid
            else:
                # Try matching by suffix
                for eid in emb_ids:
                    if eid.endswith(sid) or sid.endswith(eid):
                        matches[sid] = eid
                        break
        return matches

    train_matches = match_ids(train_scores.keys(), raw_embs.keys())
    test_matches = match_ids(test_scores.keys(), raw_embs.keys())
    print(f"  Matched: {len(train_matches)} train, {len(test_matches)} test")

    if len(train_matches) < 10 or len(test_matches) < 5:
        print("  WARNING: Too few matched proteins. Check ID format.")
        # Try looser matching
        print(f"  Score IDs (first 5): {list(train_scores.keys())[:5]}")
        print(f"  Embedding IDs (first 5): {list(raw_embs.keys())[:5]}")
        return None

    # Build training data: concatenate per-residue embeddings + Z-scores
    def build_dataset(score_dict, emb_dict, matches, skip_999=True):
        X_list, y_list = [], []
        for sid, eid in matches.items():
            scores = score_dict[sid]
            emb = emb_dict[eid]
            # Align lengths (may differ slightly due to special tokens)
            L = min(len(scores), emb.shape[0])
            s = scores[:L]
            e = emb[:L]
            # Skip 999.0 (missing values)
            if skip_999:
                mask = s != 999.0
                s = s[mask]
                e = e[mask]
            if len(s) > 0:
                X_list.append(e)
                y_list.append(s)
        if not X_list:
            return np.array([]), np.array([])
        return np.concatenate(X_list), np.concatenate(y_list)

    results = {}

    for label, emb_dict in [("raw_1024d", raw_embs), ("compressed_512d", comp_embs)]:
        print(f"\n  --- {label} ---")

        X_train, y_train = build_dataset(train_scores, emb_dict, train_matches)
        X_test, y_test = build_dataset(test_scores, emb_dict, test_matches)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"  Skipping {label}: no data")
            continue

        print(f"  Train: {X_train.shape[0]} residues, {X_train.shape[1]}d")
        print(f"  Test:  {X_test.shape[0]} residues")

        # Method 1: Ridge regression (Kibby/ADOPT style)
        t0 = time.time()
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        rho_ridge = spearmanr(y_test, y_pred_ridge).correlation
        t_ridge = time.time() - t0
        print(f"  Ridge:  Spearman ρ = {rho_ridge:.4f}  ({t_ridge:.1f}s)")

        # Method 2: Per-protein Spearman (more informative)
        per_prot_rhos = []
        for sid, eid in test_matches.items():
            scores = test_scores[sid]
            emb = emb_dict[eid]
            L = min(len(scores), emb.shape[0])
            s, e = scores[:L], emb[:L]
            mask = s != 999.0
            s, e = s[mask], e[mask]
            if len(s) > 5:
                pred = ridge.predict(e)
                rho = spearmanr(s, pred).correlation
                if np.isfinite(rho):
                    per_prot_rhos.append(rho)

        mean_rho = np.mean(per_prot_rhos) if per_prot_rhos else 0.0
        print(f"  Per-protein mean ρ = {mean_rho:.4f} (n={len(per_prot_rhos)})")

        results[label] = {
            "global_spearman": float(rho_ridge),
            "per_protein_spearman": float(mean_rho),
            "n_train_residues": int(X_train.shape[0]),
            "n_test_residues": int(X_test.shape[0]),
            "n_test_proteins": len(per_prot_rhos),
        }

    # Retention
    if "raw_1024d" in results and "compressed_512d" in results:
        retention = results["compressed_512d"]["global_spearman"] / results["raw_1024d"]["global_spearman"]
        print(f"\n  Retention: {retention:.1%} "
              f"({results['compressed_512d']['global_spearman']:.4f} / "
              f"{results['raw_1024d']['global_spearman']:.4f})")
        results["retention"] = float(retention)

    return results


# ── B. Secondary Structure Benchmark ─────────────────────────────────────

def benchmark_ss3():
    """Evaluate SS3 prediction on CB513 using LogisticRegression probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import csv

    print("\n" + "=" * 60)
    print("B. Secondary Structure (SS3, CB513)")
    print("=" * 60)

    # Load CB513 data
    cb513_path = PROJ_ROOT / "data" / "per_residue_benchmarks" / "CB513.csv"
    if not cb513_path.exists():
        print("  CB513.csv not found, skipping")
        return None

    sequences = []
    ss3_labels = []
    SS3_MAP = {"H": 0, "E": 1, "C": 2}

    with open(cb513_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row["input"])
            ss3_labels.append([SS3_MAP.get(c, 2) for c in row["dssp3"]])

    print(f"  CB513: {len(sequences)} proteins")

    # Check if we have embeddings
    emb_path = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_cb513.h5"
    if not emb_path.exists():
        print(f"  Embeddings not found at {emb_path}, skipping")
        return None

    import h5py
    raw_embs = {}
    with h5py.File(emb_path, 'r') as f:
        for key in f.keys():
            raw_embs[key] = np.array(f[key], dtype=np.float32)

    print(f"  Loaded {len(raw_embs)} embeddings")

    # Compress
    from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
    from src.one_embedding.universal_transforms import random_orthogonal_project

    corpus_h5 = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    if corpus_h5.exists():
        from src.utils.h5_store import load_residue_embeddings
        corpus = load_residue_embeddings(corpus_h5)
        stats = compute_corpus_stats(corpus, n_sample=50_000, n_pcs=5, seed=42)
    else:
        stats = compute_corpus_stats(raw_embs, n_sample=50_000, n_pcs=5, seed=42)
    top3 = stats["top_pcs"][:3]

    comp_embs = {}
    for pid, emb in raw_embs.items():
        emb_abtt = all_but_the_top(emb, top3)
        emb_rp = random_orthogonal_project(emb_abtt, d_out=512, seed=42)
        comp_embs[pid] = emb_rp.astype(np.float32)

    # Build dataset (first 400 train, last 113 test — standard CB513 split)
    keys = sorted(raw_embs.keys())
    n_train = min(400, len(keys))
    train_keys = keys[:n_train]
    test_keys = keys[n_train:]

    results = {}
    for label, emb_dict in [("raw_1024d", raw_embs), ("compressed_512d", comp_embs)]:
        # Build arrays
        X_train = np.concatenate([emb_dict[k][:len(ss3_labels[i])] for i, k in enumerate(train_keys) if k in emb_dict])
        y_train = np.concatenate([np.array(ss3_labels[i])[:emb_dict[k].shape[0]] for i, k in enumerate(train_keys) if k in emb_dict])
        X_test = np.concatenate([emb_dict[k][:len(ss3_labels[i])] for i, k in enumerate(test_keys, n_train) if k in emb_dict])
        y_test = np.concatenate([np.array(ss3_labels[i])[:emb_dict[k].shape[0]] for i, k in enumerate(test_keys, n_train) if k in emb_dict])

        lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        q3 = accuracy_score(y_test, y_pred)
        print(f"  {label}: Q3 = {q3:.4f}")
        results[label] = {"Q3": float(q3)}

    if "raw_1024d" in results and "compressed_512d" in results:
        retention = results["compressed_512d"]["Q3"] / results["raw_1024d"]["Q3"]
        print(f"  Retention: {retention:.1%}")
        results["retention"] = float(retention)

    return results


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 36: Toolkit benchmarks")
    parser.add_argument("--task", type=str, default="all",
                        choices=["disorder", "ss3", "all"],
                        help="Which benchmark to run")
    args = parser.parse_args()

    RESULTS_PATH = PROJ_ROOT / "data" / "benchmarks" / "toolkit_benchmark_results.json"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.task in ("disorder", "all"):
        r = benchmark_disorder()
        if r:
            all_results["disorder"] = r

    if args.task in ("ss3", "all"):
        r = benchmark_ss3()
        if r:
            all_results["ss3"] = r

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved: {RESULTS_PATH}")
