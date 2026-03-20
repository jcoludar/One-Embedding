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
    import random

    print("\n" + "=" * 60)
    print("B. Secondary Structure (SS3, CB513)")
    print("=" * 60)

    # Load CB513 data using the canonical loader (returns dicts keyed by protein ID)
    from src.evaluation.per_residue_tasks import load_cb513_csv
    _, ss3_labels, _, _ = load_cb513_csv(
        PROJ_ROOT / "data" / "per_residue_benchmarks" / "CB513.csv"
    )
    if not ss3_labels:
        print("  CB513.csv not found or empty, skipping")
        return None

    print(f"  CB513: {len(ss3_labels)} proteins with SS3 labels")

    SS3_MAP = {"H": 0, "E": 1, "C": 2}

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

    # Build dataset: 80/20 split by protein ID (seed=42), matching Exp 39 methodology
    # Use only proteins that have BOTH embeddings and labels
    common_ids = sorted(pid for pid in ss3_labels if pid in raw_embs)
    rng = random.Random(42)
    rng.shuffle(common_ids)
    split_idx = int(0.8 * len(common_ids))
    train_keys = common_ids[:split_idx]
    test_keys = common_ids[split_idx:]
    print(f"  Split: {len(train_keys)} train, {len(test_keys)} test (seed=42)")

    results = {}
    for label, emb_dict in [("raw_1024d", raw_embs), ("compressed_512d", comp_embs)]:
        # Build arrays using dict-based lookup (no positional indexing)
        X_train_parts, y_train_parts = [], []
        for pid in train_keys:
            emb = emb_dict[pid]
            lab_str = ss3_labels[pid]
            lab = np.array([SS3_MAP[c] for c in lab_str], dtype=np.int64)
            L = min(emb.shape[0], len(lab))
            X_train_parts.append(emb[:L])
            y_train_parts.append(lab[:L])

        X_test_parts, y_test_parts = [], []
        for pid in test_keys:
            emb = emb_dict[pid]
            lab_str = ss3_labels[pid]
            lab = np.array([SS3_MAP[c] for c in lab_str], dtype=np.int64)
            L = min(emb.shape[0], len(lab))
            X_test_parts.append(emb[:L])
            y_test_parts.append(lab[:L])

        X_train = np.concatenate(X_train_parts)
        y_train = np.concatenate(y_train_parts)
        X_test = np.concatenate(X_test_parts)
        y_test = np.concatenate(y_test_parts)

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


# ── C. Family Classification (Ret@1) ─────────────────────────────────────

def benchmark_classifier():
    """Evaluate family retrieval on SCOPe 5K: raw vs compressed protein vectors."""
    import csv as csv_mod
    from src.evaluation.retrieval import evaluate_retrieval_from_vectors

    print("\n" + "=" * 60)
    print("C. Family Classification (SCOPe 5K Ret@1)")
    print("=" * 60)

    EMB_PATH = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    META_PATH = PROJ_ROOT / "data" / "proteins" / "metadata_5k.csv"

    if not EMB_PATH.exists() or not META_PATH.exists():
        print("  Missing embeddings or metadata, skipping")
        return None

    # Load metadata
    metadata = []
    with open(META_PATH) as f:
        for row in csv_mod.DictReader(f):
            metadata.append(row)
    print(f"  Metadata: {len(metadata)} proteins")

    # Load and compress
    raw_embs, comp_embs = load_embeddings_and_compress(EMB_PATH)

    # Pool to protein vectors (mean pool)
    raw_vecs = {pid: emb.mean(axis=0) for pid, emb in raw_embs.items()}
    comp_vecs = {pid: emb.mean(axis=0) for pid, emb in comp_embs.items()}

    results = {}
    for label, vecs in [("raw_1024d", raw_vecs), ("compressed_512d", comp_vecs)]:
        r = evaluate_retrieval_from_vectors(vecs, metadata, label_key="family")
        ret1 = r.get("precision@1", 0.0)
        mrr = r.get("mrr", 0.0)
        print(f"  {label}: Ret@1={ret1:.4f}, MRR={mrr:.4f}")
        results[label] = {"ret_at_1": float(ret1), "mrr": float(mrr)}

    if "raw_1024d" in results and "compressed_512d" in results:
        retention = results["compressed_512d"]["ret_at_1"] / results["raw_1024d"]["ret_at_1"]
        print(f"  Ret@1 Retention: {retention:.1%}")
        results["retention"] = float(retention)

    return results


# ── D. TM-score Correlation ───────────────────────────────────────────────

def benchmark_tm_score():
    """Evaluate embedding distance vs structural TM-score correlation."""
    from scipy.stats import spearmanr, pearsonr

    print("\n" + "=" * 60)
    print("D. TM-score Correlation (200 proteins)")
    print("=" * 60)

    TM_PATH = PROJ_ROOT / "data" / "structures" / "tm_scores_200.npz"
    EMB_PATH = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_medium5k.h5"

    if not TM_PATH.exists() or not EMB_PATH.exists():
        print("  Missing TM-score data or embeddings, skipping")
        return None

    # Load TM-scores
    tm_data = np.load(TM_PATH, allow_pickle=True)
    tm_scores = tm_data["tm_scores"]  # (200, 200)
    tm_ids = list(tm_data["protein_ids"])
    print(f"  TM-scores: {len(tm_ids)} proteins")

    # Load and compress embeddings for these proteins
    raw_embs, comp_embs = load_embeddings_and_compress(EMB_PATH, set(tm_ids))

    # Mean pool to protein vectors
    raw_vecs = {pid: emb.mean(axis=0) for pid, emb in raw_embs.items() if pid in tm_ids}
    comp_vecs = {pid: emb.mean(axis=0) for pid, emb in comp_embs.items() if pid in tm_ids}

    # Filter to proteins we have both embeddings and TM-scores for
    common_ids = [pid for pid in tm_ids if pid in raw_vecs]
    idx_map = {pid: i for i, pid in enumerate(tm_ids)}
    print(f"  Matched: {len(common_ids)} proteins")

    if len(common_ids) < 20:
        print("  Too few matched proteins, skipping")
        return None

    results = {}
    for label, vecs in [("raw_1024d", raw_vecs), ("compressed_512d", comp_vecs)]:
        # Build cosine similarity matrix
        ids = [pid for pid in common_ids if pid in vecs]
        V = np.array([vecs[pid] for pid in ids])
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_norm = V / (norms + 1e-10)
        cos_sim = V_norm @ V_norm.T

        # Extract corresponding TM-scores
        tm_sub = np.array([[tm_scores[idx_map[a], idx_map[b]] for b in ids] for a in ids])

        # Upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
        rho = spearmanr(tm_sub[mask], cos_sim[mask]).correlation
        r = pearsonr(tm_sub[mask], cos_sim[mask])[0]

        print(f"  {label}: Spearman ρ={rho:.4f}, Pearson r={r:.4f}")
        results[label] = {"spearman": float(rho), "pearson": float(r)}

    if "raw_1024d" in results and "compressed_512d" in results:
        retention = results["compressed_512d"]["spearman"] / results["raw_1024d"]["spearman"]
        print(f"  Spearman Retention: {retention:.1%}")
        results["retention"] = float(retention)

    return results


# ── E. Conservation Scoring ───────────────────────────────────────────────

def benchmark_conservation():
    """Evaluate embedding-derived conservation vs variance-based proxy."""

    print("\n" + "=" * 60)
    print("E. Conservation Scoring (variance proxy)")
    print("=" * 60)

    EMB_PATH = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    META_PATH = PROJ_ROOT / "data" / "proteins" / "metadata_5k.csv"

    if not EMB_PATH.exists():
        print("  Missing embeddings, skipping")
        return None

    import csv as csv_mod
    # Group proteins by superfamily for conservation estimation
    meta = {}
    with open(META_PATH) as f:
        for row in csv_mod.DictReader(f):
            meta[row["id"]] = row

    raw_embs, comp_embs = load_embeddings_and_compress(EMB_PATH)

    # Group by superfamily (need ≥3 members)
    from collections import defaultdict
    sf_groups = defaultdict(list)
    for pid in raw_embs:
        if pid in meta:
            sf_groups[meta[pid]["superfamily"]].append(pid)

    valid_sfs = {sf: pids for sf, pids in sf_groups.items() if len(pids) >= 3}
    print(f"  Superfamilies with ≥3 members: {len(valid_sfs)}")

    # For each superfamily: compute per-residue embedding variance (conservation proxy)
    # across family members (mean-pooled). Higher variance = less conserved.
    # Compare raw vs compressed variance patterns.
    from scipy.stats import spearmanr

    rhos = []
    for sf, pids in list(valid_sfs.items())[:50]:  # cap at 50 superfamilies
        # Pool to protein vectors
        raw_vecs = np.array([raw_embs[pid].mean(axis=0) for pid in pids])
        comp_vecs = np.array([comp_embs[pid].mean(axis=0) for pid in pids])

        # Per-dimension variance across family members
        var_raw = np.var(raw_vecs, axis=0)  # (1024,) or (512,)
        var_comp = np.var(comp_vecs, axis=0)

        # We can't directly compare dimensions (different spaces).
        # Instead: compute pairwise distance matrix and compare.
        from scipy.spatial.distance import pdist
        d_raw = pdist(raw_vecs, metric="cosine")
        d_comp = pdist(comp_vecs, metric="cosine")

        if len(d_raw) > 3:
            rho = spearmanr(d_raw, d_comp).correlation
            if np.isfinite(rho):
                rhos.append(rho)

    if not rhos:
        print("  No valid superfamilies, skipping")
        return None

    mean_rho = np.mean(rhos)
    print(f"  Pairwise distance correlation (raw vs compressed): ρ={mean_rho:.4f} ± {np.std(rhos):.4f}")
    print(f"  n={len(rhos)} superfamilies")

    results = {
        "pairwise_distance_spearman": float(mean_rho),
        "n_superfamilies": len(rhos),
    }
    return results


# ── F. Alignment Quality ─────────────────────────────────────────────────

def benchmark_aligner():
    """Evaluate embedding-based alignment consistency: raw vs compressed."""
    from src.one_embedding.aligner import align_embeddings

    print("\n" + "=" * 60)
    print("F. Alignment Consistency (raw vs compressed)")
    print("=" * 60)

    EMB_PATH = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    META_PATH = PROJ_ROOT / "data" / "proteins" / "metadata_5k.csv"

    if not EMB_PATH.exists():
        print("  Missing embeddings, skipping")
        return None

    import csv as csv_mod
    meta = {}
    with open(META_PATH) as f:
        for row in csv_mod.DictReader(f):
            meta[row["id"]] = row

    raw_embs, comp_embs = load_embeddings_and_compress(EMB_PATH)

    # Pick pairs within same family for alignment
    from collections import defaultdict
    fam_groups = defaultdict(list)
    for pid in raw_embs:
        if pid in meta:
            fam_groups[meta[pid]["family"]].append(pid)

    # Select 50 within-family pairs + 50 between-family pairs
    pairs = []
    for fam, pids in fam_groups.items():
        if len(pids) >= 2:
            pairs.append((pids[0], pids[1], "within"))
            if len(pairs) >= 50:
                break

    # For each pair: align with raw and compressed, compare alignment score & overlap
    scores_raw, scores_comp = [], []
    alignment_overlaps = []

    for pid_a, pid_b, rel in pairs:
        r_raw = align_embeddings(raw_embs[pid_a], raw_embs[pid_b], mode="global")
        r_comp = align_embeddings(comp_embs[pid_a], comp_embs[pid_b], mode="global")

        scores_raw.append(r_raw["score"])
        scores_comp.append(r_comp["score"])

        # Alignment overlap: fraction of positions that agree
        a_raw = set(zip(r_raw["align_a"], r_raw["align_b"]))
        a_comp = set(zip(r_comp["align_a"], r_comp["align_b"]))
        overlap = len(a_raw & a_comp) / max(len(a_raw), 1)
        alignment_overlaps.append(overlap)

    if not scores_raw:
        print("  No valid pairs, skipping")
        return None

    from scipy.stats import spearmanr
    score_rho = spearmanr(scores_raw, scores_comp).correlation

    print(f"  {len(pairs)} within-family pairs aligned")
    print(f"  Score correlation (raw vs compressed): ρ={score_rho:.4f}")
    print(f"  Alignment overlap: {np.mean(alignment_overlaps):.1%} ± {np.std(alignment_overlaps):.1%}")

    results = {
        "n_pairs": len(pairs),
        "score_spearman": float(score_rho),
        "alignment_overlap_mean": float(np.mean(alignment_overlaps)),
        "alignment_overlap_std": float(np.std(alignment_overlaps)),
    }
    return results


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 36: Toolkit benchmarks")
    parser.add_argument("--task", type=str, default="all",
                        choices=["disorder", "ss3", "classifier", "tm_score",
                                 "conservation", "aligner", "all"],
                        help="Which benchmark to run")
    args = parser.parse_args()

    RESULTS_PATH = PROJ_ROOT / "data" / "benchmarks" / "toolkit_benchmark_results.json"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results to accumulate
    all_results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)

    tasks = {
        "disorder": benchmark_disorder,
        "ss3": benchmark_ss3,
        "classifier": benchmark_classifier,
        "tm_score": benchmark_tm_score,
        "conservation": benchmark_conservation,
        "aligner": benchmark_aligner,
    }

    to_run = list(tasks.keys()) if args.task == "all" else [args.task]

    for task_name in to_run:
        r = tasks[task_name]()
        if r:
            all_results[task_name] = r

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved: {RESULTS_PATH}")
