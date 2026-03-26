#!/usr/bin/env python3
"""ABTT Cross-Corpus Stability Test — Novel Methodological Contribution.

Proves that the top-3 PCs removed by ABTT are properties of the ProtT5-XL
architecture, not properties of any specific dataset. This closes the
SCOPe-on-SCOPe overlap question: if the PCs are the same regardless of
fitting corpus, the choice doesn't matter.

Corpora tested:
    1. SCOPe 5K (2493 proteins) — the fitting corpus used in Exp 43
    2. CB513 (513 proteins) — secondary structure benchmark
    3. CheZOD (1174 proteins) — disorder benchmark
    4. Tiny subset (50 random SCOPe proteins) — stability under subsampling

References:
    Bjorck & Golub (1973) principal angles between subspaces.
    Mu & Viswanath (2018) All-But-The-Top.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import h5py

# Project paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics.abtt_stability import cross_corpus_stability_report, principal_angles, subspace_similarity
from config import RAW_EMBEDDINGS, RESULTS_DIR


def load_residues_from_h5(path: Path, max_proteins: int = None) -> np.ndarray:
    """Load and stack all per-residue embeddings from an H5 file."""
    parts = []
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if max_proteins:
            keys = keys[:max_proteins]
        for key in keys:
            emb = f[key][:]
            parts.append(emb.astype(np.float32))
    return np.vstack(parts)


def main():
    print("=" * 70)
    print("ABTT Cross-Corpus Stability Test")
    print("=" * 70)
    print()

    t0 = time.time()

    # Load corpora
    print("Loading corpora...")

    scope_path = RAW_EMBEDDINGS["prot_t5"]
    cb513_path = RAW_EMBEDDINGS["prot_t5_cb513"]
    chezod_path = RAW_EMBEDDINGS["prot_t5_chezod"]

    corpora = {}

    print(f"  SCOPe 5K: {scope_path}")
    corpora["SCOPe_5K"] = load_residues_from_h5(scope_path)
    print(f"    → {corpora['SCOPe_5K'].shape[0]:,} residues, D={corpora['SCOPe_5K'].shape[1]}")

    print(f"  CB513: {cb513_path}")
    corpora["CB513"] = load_residues_from_h5(cb513_path)
    print(f"    → {corpora['CB513'].shape[0]:,} residues, D={corpora['CB513'].shape[1]}")

    print(f"  CheZOD: {chezod_path}")
    corpora["CheZOD"] = load_residues_from_h5(chezod_path)
    print(f"    → {corpora['CheZOD'].shape[0]:,} residues, D={corpora['CheZOD'].shape[1]}")

    # Tiny subset: first 50 proteins from SCOPe
    print(f"  Tiny_50: first 50 proteins from SCOPe 5K")
    corpora["Tiny_50"] = load_residues_from_h5(scope_path, max_proteins=50)
    print(f"    → {corpora['Tiny_50'].shape[0]:,} residues, D={corpora['Tiny_50'].shape[1]}")

    print(f"\nLoading took {time.time() - t0:.1f}s")
    print()

    # Run stability analysis
    print("Fitting ABTT on each corpus and comparing top-3 PCs...")
    print()

    t1 = time.time()
    report = cross_corpus_stability_report(corpora, k=3, seed=42)
    print(f"Analysis took {time.time() - t1:.1f}s")
    print()

    # Also fit ABTT separately to get mean vectors + explained variance
    from src.one_embedding.core.preprocessing import fit_abtt
    fitted_params = {}
    for name, data in corpora.items():
        fitted_params[name] = fit_abtt(data, k=3, seed=42)

    # Print results
    print("-" * 70)
    print("PAIRWISE SUBSPACE SIMILARITY (1.0 = identical, 0.0 = orthogonal)")
    print("-" * 70)

    for (a, b), sim in sorted(report["pairwise_similarity"].items()):
        angles = report["pairwise_angles_deg"][(a, b)]
        angles_str = ", ".join(f"{a:.2f}°" for a in angles)
        status = "✓" if sim > 0.95 else "⚠"
        print(f"  {status} {a:>10s} vs {b:<10s}  sim={sim:.6f}  angles=[{angles_str}]")

    print()
    print(f"  Min similarity: {report['min_similarity']:.6f}")
    print(f"  Mean similarity: {report['mean_similarity']:.6f}")
    print(f"  Conclusion: {report['conclusion'].upper()}")
    print()

    # Interpretation
    if report["conclusion"] == "stable":
        print("INTERPRETATION: The top-3 PCs are STABLE across all corpora.")
        print("  → They are properties of the ProtT5-XL architecture, not the data.")
        print("  → ABTT fitting corpus choice is irrelevant (within noise).")
        print("  → The SCOPe-on-SCOPe overlap in Exp 43 retrieval benchmarks")
        print("    does NOT constitute information leakage.")
    else:
        print("WARNING: PCs show INSTABILITY across corpora.")
        print("  → The ABTT fitting corpus matters — further investigation needed.")
        print("  → Consider using a dedicated external corpus for all benchmarks.")

    print()

    # Mean vector similarity
    print("-" * 70)
    print("MEAN VECTOR SIMILARITY (cosine)")
    print("-" * 70)
    names = sorted(corpora.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mean_a = fitted_params[a]["mean"]
            mean_b = fitted_params[b]["mean"]
            cos_sim = float(np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b)))
            print(f"  {a:>10s} vs {b:<10s}  cosine={cos_sim:.6f}")
    print()

    # Per-PC analysis: check which PCs are stable vs unstable
    print("-" * 70)
    print("PER-PC ANALYSIS: Individual PC direction cosines")
    print("-" * 70)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pcs_a = fitted_params[a]["top_pcs"]  # (3, D)
            pcs_b = fitted_params[b]["top_pcs"]  # (3, D)
            print(f"  {a} vs {b}:")
            for pc_idx in range(3):
                cos = abs(float(np.dot(pcs_a[pc_idx], pcs_b[pc_idx])))
                print(f"    PC{pc_idx+1}: |cos|={cos:.6f} ({np.degrees(np.arccos(min(cos, 1.0))):.1f}°)")
            print()

    # Explained variance analysis
    print("-" * 70)
    print("EXPLAINED VARIANCE (top-3 PCs as fraction of total)")
    print("-" * 70)
    for name, data in corpora.items():
        sample = data
        if len(data) > 50000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(data), 50000, replace=False)
            sample = data[idx]
        centered = sample - sample.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var_explained = s**2 / np.sum(s**2)
        total_top3 = float(np.sum(var_explained[:3]))
        print(f"  {name:>10s}: PC1={var_explained[0]:.4f}, PC2={var_explained[1]:.4f}, PC3={var_explained[2]:.4f}, total_top3={total_top3:.4f}")
    print()

    # Downstream performance test: does it matter for retrieval?
    print("-" * 70)
    print("DOWNSTREAM IMPACT: Retrieval Ret@1 with ABTT fitted on different corpora")
    print("-" * 70)
    print()

    from src.one_embedding.core.preprocessing import apply_abtt
    from src.one_embedding.core.projection import project
    from scipy.fft import dct

    # Load SCOPe 5K per-protein embeddings
    scope_embeddings = {}
    with h5py.File(scope_path, "r") as f:
        for key in list(f.keys()):
            scope_embeddings[key] = f[key][:].astype(np.float32)

    # Load metadata for family labels
    import csv
    metadata_path = ROOT / "data" / "proteins" / "metadata_5k.csv"
    id_to_family = {}
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_family[row["id"]] = row["family"]

    def encode_and_retrieval(abtt_params, corpus_name):
        """Encode SCOPe proteins with given ABTT params, compute Ret@1."""
        vectors = {}
        for pid, emb in scope_embeddings.items():
            # Apply ABTT
            processed = apply_abtt(emb, abtt_params)
            # Random projection to 768d
            projected = project(processed, d_out=768, seed=42)
            # DCT K=4 protein vector
            coeffs = dct(projected, axis=0, type=2, norm="ortho")[:4]
            vectors[pid] = coeffs.flatten().astype(np.float32)

        # Compute Ret@1 (cosine)
        pids = [pid for pid in vectors if pid in id_to_family]
        mat = np.array([vectors[pid] for pid in pids], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        mat_normed = mat / norms
        sims = mat_normed @ mat_normed.T

        correct = 0
        for i in range(len(pids)):
            sims_row = sims[i].copy()
            sims_row[i] = -np.inf
            top1 = np.argmax(sims_row)
            if id_to_family[pids[top1]] == id_to_family[pids[i]]:
                correct += 1

        ret1 = correct / len(pids)
        return ret1, len(pids)

    # Test with each corpus's ABTT params
    downstream_results = {}
    for name in sorted(fitted_params.keys()):
        ret1, n = encode_and_retrieval(fitted_params[name], name)
        downstream_results[name] = ret1
        print(f"  ABTT fitted on {name:>10s}: Ret@1 = {ret1:.4f} (n={n})")

    # Also test with no ABTT (raw)
    raw_vectors = {}
    for pid, emb in scope_embeddings.items():
        projected = project(emb, d_out=768, seed=42)
        coeffs = dct(projected, axis=0, type=2, norm="ortho")[:4]
        raw_vectors[pid] = coeffs.flatten().astype(np.float32)

    pids = [pid for pid in raw_vectors if pid in id_to_family]
    mat = np.array([raw_vectors[pid] for pid in pids], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_normed = mat / np.maximum(norms, 1e-10)
    sims = mat_normed @ mat_normed.T
    correct = sum(1 for i in range(len(pids)) if id_to_family[pids[np.argmax(np.where(np.arange(len(pids)) != i, sims[i], -np.inf))]] == id_to_family[pids[i]])
    raw_ret1 = correct / len(pids)
    print(f"  No ABTT (raw RP768): Ret@1 = {raw_ret1:.4f} (n={len(pids)})")

    vals = list(downstream_results.values())
    max_diff = max(vals) - min(vals)
    print(f"\n  Max difference across corpora: {max_diff:.4f} ({max_diff*100:.2f}pp)")
    print(f"  ABTT vs raw improvement range: {min(vals) - raw_ret1:.4f} to {max(vals) - raw_ret1:.4f}")

    if max_diff < 0.005:
        print("\n  CONCLUSION: Downstream performance is INSENSITIVE to ABTT fitting corpus")
        print("  (max difference < 0.5pp across all corpora, including 50-protein subset)")
        downstream_conclusion = "insensitive"
    else:
        print(f"\n  CONCLUSION: Fitting corpus affects performance by {max_diff*100:.1f}pp")
        downstream_conclusion = "sensitive"
    print()

    # Save results
    results_path = RESULTS_DIR / "abtt_stability_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert tuple keys to strings for JSON
    results = {
        "pairwise_similarity": {f"{a}_vs_{b}": sim for (a, b), sim in report["pairwise_similarity"].items()},
        "pairwise_angles_deg": {f"{a}_vs_{b}": angles for (a, b), angles in report["pairwise_angles_deg"].items()},
        "min_similarity": report["min_similarity"],
        "mean_similarity": report["mean_similarity"],
        "conclusion": report["conclusion"],
        "corpora_shapes": {name: list(corpora[name].shape) for name in corpora},
        "mean_cosine_similarities": {},
        "downstream_ret1": downstream_results,
        "downstream_ret1_no_abtt": raw_ret1,
        "downstream_max_diff": max_diff,
        "downstream_conclusion": downstream_conclusion,
        "_meta": {
            "script": "run_abtt_stability.py",
            "k": 3,
            "seed": 42,
            "threshold": 0.95,
        },
    }

    # Add mean vector cosine similarities
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mean_a = fitted_params[a]["mean"]
            mean_b = fitted_params[b]["mean"]
            cos_sim = float(np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b)))
            results["mean_cosine_similarities"][f"{a}_vs_{b}"] = cos_sim

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
