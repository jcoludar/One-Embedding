#!/usr/bin/env python3
"""Experiment 37 — Structural Information Retention Benchmark.

Tests whether codec-compressed embeddings preserve structural distance patterns
by comparing embedding-derived residue-pair distances against Cα distances
from PDB structures. Uses lDDT scoring (borrowed from EMBER3D) and contact
prediction precision.

Pipeline:
  1. Load ProtT5 per-residue embeddings (raw 1024d, compressed 512d)
  2. Download PDB structures for SCOPe domains
  3. Compute Cα distance matrices (ground truth)
  4. Compute embedding distance matrices (raw and compressed)
  5. Score: lDDT, contact precision, Spearman correlation

Usage:
  uv run python experiments/37_structural_retention.py --n-proteins 50
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))


# ── lDDT (from EMBER3D, adapted to numpy) ───────────────────────────────

def lddt_numpy(dist_pred: np.ndarray, dist_gt: np.ndarray) -> np.ndarray:
    """Local Distance Difference Test — per-residue structural quality score.

    Adapted from EMBER3D (Weissenow et al.) to pure numpy.

    Args:
        dist_pred: (L, L) predicted distance matrix
        dist_gt: (L, L) ground truth distance matrix (Å)

    Returns:
        (L,) per-residue lDDT scores in [0, 1]
    """
    L = dist_gt.shape[0]
    cutoff = 15.0
    mask = (dist_gt < cutoff) * (1.0 - np.eye(L))

    dist_l1 = np.abs(dist_pred - dist_gt)

    score = ((dist_l1 < 0.5).astype(np.float64) +
             (dist_l1 < 1.0).astype(np.float64) +
             (dist_l1 < 2.0).astype(np.float64) +
             (dist_l1 < 4.0).astype(np.float64)) * 0.25

    score *= mask

    per_residue = (1e-10 + np.sum(score, axis=-1)) / (1e-10 + np.sum(mask, axis=-1))
    return per_residue


def contact_precision(dist_emb: np.ndarray, dist_gt: np.ndarray,
                      contact_thresh: float = 8.0,
                      top_k_factor: float = 0.2,
                      min_seq_sep: int = 6) -> float:
    """Precision of top-L*k predicted contacts.

    Args:
        dist_emb: (L, L) embedding-derived distance matrix
        dist_gt: (L, L) Cα distance matrix (Å)
        contact_thresh: Å cutoff for true contacts
        top_k_factor: fraction of L to consider (L/5 = 0.2)
        min_seq_sep: minimum sequence separation for contacts

    Returns:
        Precision in [0, 1]
    """
    L = dist_gt.shape[0]
    k = max(1, int(L * top_k_factor))

    # True contacts
    contacts_gt = (dist_gt < contact_thresh).astype(float)

    # Apply sequence separation mask
    sep_mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= min_seq_sep
    contacts_gt *= sep_mask

    # Predicted: smallest embedding distances = most likely contacts
    dist_masked = dist_emb.copy()
    dist_masked[~sep_mask] = np.inf
    np.fill_diagonal(dist_masked, np.inf)

    # Top k predictions (smallest distances)
    flat_idx = np.argsort(dist_masked.ravel())[:k]
    rows, cols = np.unravel_index(flat_idx, (L, L))

    # Precision
    n_correct = sum(contacts_gt[r, c] for r, c in zip(rows, cols))
    return float(n_correct / k)


# ── PDB structure fetching ───────────────────────────────────────────────

def fetch_ca_coords(pdb_id: str, chain_id: str = "A",
                    cache_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """Fetch Cα coordinates from RCSB PDB.

    Returns (L, 3) array of Cα positions in Å, or None on failure.
    """
    from Bio.PDB import PDBParser, MMCIFParser
    import urllib.request
    import gzip
    import tempfile

    if cache_dir is None:
        cache_dir = PROJ_ROOT / "data" / "pdb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdb_id = pdb_id.lower()
    cache_path = cache_dir / f"{pdb_id}.cif.gz"

    # Check cache size (cap at ~500 MB to be disk-friendly)
    if not cache_path.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file())
        if cache_size > 500 * 1024 * 1024:  # 500 MB cap
            return None  # stop downloading

        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        try:
            urllib.request.urlretrieve(url, str(cache_path))
        except Exception:
            return None

    # Parse
    try:
        with gzip.open(str(cache_path), "rt") as f:
            tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False)
            tmpfile.write(f.read())
            tmpfile.close()

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, tmpfile.name)
        Path(tmpfile.name).unlink()

        model = structure[0]

        # Find the right chain
        chain = None
        for c in model.get_chains():
            if c.id.upper() == chain_id.upper():
                chain = c
                break
        if chain is None:
            # Try first chain
            chain = list(model.get_chains())[0]

        # Extract Cα coordinates
        ca_coords = []
        for residue in chain.get_residues():
            if residue.id[0] != " ":  # skip heteroatoms
                continue
            if "CA" in residue:
                ca_coords.append(residue["CA"].get_vector().get_array())

        if len(ca_coords) < 10:
            return None

        return np.array(ca_coords, dtype=np.float64)

    except Exception:
        return None


def scop_to_pdb(domain_id: str) -> Tuple[str, str]:
    """Convert SCOPe domain ID to PDB ID + chain.

    e.g., 'd1a0aa_' → ('1a0a', 'A'), 'd1dlwa1' → ('1dlw', 'A')
    """
    # Format: d<pdb_id><chain><domain>
    # e.g., d1a0aa_ → pdb=1a0a, chain=a
    m = re.match(r"d(\w{4})(\w)", domain_id)
    if m:
        pdb_id = m.group(1)
        chain = m.group(2).upper()
        if chain == "_":
            chain = "A"
        return pdb_id, chain
    return "", ""


# ── Main benchmark ───────────────────────────────────────────────────────

def run_benchmark(n_proteins: int = 50, seed: int = 42):
    """Run structural retention benchmark on SCOPe 5K proteins."""
    import h5py
    from scipy.stats import spearmanr
    from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
    from src.one_embedding.universal_transforms import random_orthogonal_project
    from src.utils.h5_store import load_residue_embeddings

    print("=" * 60)
    print("Experiment 37: Structural Information Retention")
    print("=" * 60)

    EMB_PATH = PROJ_ROOT / "data" / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    if not EMB_PATH.exists():
        print("ERROR: ProtT5 SCOPe embeddings not found")
        return None

    # Load embeddings
    print("\n  Loading embeddings...")
    raw_embs = {}
    with h5py.File(EMB_PATH, "r") as f:
        for key in f.keys():
            raw_embs[key] = np.array(f[key], dtype=np.float32)
    print(f"  Loaded {len(raw_embs)} proteins")

    # ABTT3 + RP512 compression
    print("  Computing ABTT3 stats...")
    corpus = load_residue_embeddings(EMB_PATH)
    stats = compute_corpus_stats(corpus, n_sample=50_000, n_pcs=5, seed=42)
    top3 = stats["top_pcs"][:3]

    comp_embs = {}
    for pid, emb in raw_embs.items():
        emb_abtt = all_but_the_top(emb, top3)
        emb_rp = random_orthogonal_project(emb_abtt, d_out=512, seed=42)
        comp_embs[pid] = emb_rp.astype(np.float32)

    print(f"  Compressed: 1024d → 512d")

    # Select proteins with PDB structures
    print(f"\n  Fetching PDB structures for up to {n_proteins} proteins...")
    rng = np.random.RandomState(seed)
    domain_ids = sorted(raw_embs.keys())
    rng.shuffle(domain_ids)

    results = []
    n_fetched = 0
    n_tried = 0

    for domain_id in domain_ids:
        if n_fetched >= n_proteins:
            break
        n_tried += 1

        pdb_id, chain = scop_to_pdb(domain_id)
        if not pdb_id:
            continue

        emb_raw = raw_embs[domain_id]
        emb_comp = comp_embs[domain_id]
        L_emb = emb_raw.shape[0]

        # Skip very short or very long proteins
        if L_emb < 30 or L_emb > 500:
            continue

        ca_coords = fetch_ca_coords(pdb_id, chain)
        if ca_coords is None:
            continue

        L_pdb = ca_coords.shape[0]

        # Align lengths (PDB and embedding may differ by a few residues)
        L = min(L_emb, L_pdb)
        if abs(L_emb - L_pdb) > 10:
            continue  # too much mismatch
        emb_raw = emb_raw[:L]
        emb_comp = emb_comp[:L]
        ca_coords = ca_coords[:L]

        # Compute distance matrices
        # Structural (Cα-Cα)
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        dist_struct = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Embedding-derived (Euclidean in embedding space)
        diff_raw = emb_raw[:, None, :] - emb_raw[None, :, :]
        dist_raw = np.sqrt(np.sum(diff_raw.astype(np.float64) ** 2, axis=-1))

        diff_comp = emb_comp[:, None, :] - emb_comp[None, :, :]
        dist_comp = np.sqrt(np.sum(diff_comp.astype(np.float64) ** 2, axis=-1))

        # Scale embedding distances to Å range for lDDT
        # Use linear scaling: dist_Å ≈ dist_emb * (median_struct / median_emb)
        med_struct = np.median(dist_struct[dist_struct > 0])
        med_raw = np.median(dist_raw[dist_raw > 0])
        med_comp = np.median(dist_comp[dist_comp > 0])

        dist_raw_scaled = dist_raw * (med_struct / med_raw) if med_raw > 0 else dist_raw
        dist_comp_scaled = dist_comp * (med_struct / med_comp) if med_comp > 0 else dist_comp

        # lDDT
        lddt_raw = lddt_numpy(dist_raw_scaled, dist_struct)
        lddt_comp = lddt_numpy(dist_comp_scaled, dist_struct)

        # Contact precision (L/5)
        cp_raw = contact_precision(dist_raw, dist_struct)
        cp_comp = contact_precision(dist_comp, dist_struct)

        # Spearman correlation of upper triangle
        mask = np.triu(np.ones((L, L), dtype=bool), k=1)
        rho_raw = spearmanr(dist_struct[mask], dist_raw[mask]).correlation
        rho_comp = spearmanr(dist_struct[mask], dist_comp[mask]).correlation

        result = {
            "domain": domain_id,
            "pdb": pdb_id,
            "chain": chain,
            "length": L,
            "lddt_raw": float(np.mean(lddt_raw)),
            "lddt_comp": float(np.mean(lddt_comp)),
            "lddt_retention": float(np.mean(lddt_comp) / np.mean(lddt_raw)) if np.mean(lddt_raw) > 0 else 0,
            "contact_prec_raw": cp_raw,
            "contact_prec_comp": cp_comp,
            "contact_retention": cp_comp / cp_raw if cp_raw > 0 else 0,
            "spearman_raw": float(rho_raw),
            "spearman_comp": float(rho_comp),
            "spearman_retention": float(rho_comp / rho_raw) if rho_raw > 0 else 0,
        }
        results.append(result)
        n_fetched += 1

        if n_fetched % 10 == 0:
            print(f"  [{n_fetched}/{n_proteins}] {domain_id} (PDB:{pdb_id}) L={L} "
                  f"lDDT={np.mean(lddt_raw):.3f}→{np.mean(lddt_comp):.3f} "
                  f"CP={cp_raw:.3f}→{cp_comp:.3f} "
                  f"ρ={rho_raw:.3f}→{rho_comp:.3f}")

    print(f"\n  Fetched structures for {n_fetched}/{n_tried} attempted proteins")

    if not results:
        print("  No proteins with matching structures found!")
        return None

    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS: Structural Information Retention")
    print("=" * 60)

    lddt_raw_arr = np.array([r["lddt_raw"] for r in results])
    lddt_comp_arr = np.array([r["lddt_comp"] for r in results])
    cp_raw_arr = np.array([r["contact_prec_raw"] for r in results])
    cp_comp_arr = np.array([r["contact_prec_comp"] for r in results])
    rho_raw_arr = np.array([r["spearman_raw"] for r in results])
    rho_comp_arr = np.array([r["spearman_comp"] for r in results])

    print(f"\n  {len(results)} proteins evaluated")
    print(f"\n  lDDT (embedding distances vs Cα distances):")
    print(f"    Raw 1024d:      {np.mean(lddt_raw_arr):.4f} ± {np.std(lddt_raw_arr):.4f}")
    print(f"    Compressed 512d: {np.mean(lddt_comp_arr):.4f} ± {np.std(lddt_comp_arr):.4f}")
    print(f"    Retention:       {np.mean(lddt_comp_arr)/np.mean(lddt_raw_arr):.1%}")

    print(f"\n  Contact Precision (top L/5, 8Å, seq_sep≥6):")
    print(f"    Raw 1024d:      {np.mean(cp_raw_arr):.4f} ± {np.std(cp_raw_arr):.4f}")
    print(f"    Compressed 512d: {np.mean(cp_comp_arr):.4f} ± {np.std(cp_comp_arr):.4f}")
    print(f"    Retention:       {np.mean(cp_comp_arr)/np.mean(cp_raw_arr):.1%}")

    print(f"\n  Spearman ρ (embedding vs structural distances):")
    print(f"    Raw 1024d:      {np.mean(rho_raw_arr):.4f} ± {np.std(rho_raw_arr):.4f}")
    print(f"    Compressed 512d: {np.mean(rho_comp_arr):.4f} ± {np.std(rho_comp_arr):.4f}")
    print(f"    Retention:       {np.mean(rho_comp_arr)/np.mean(rho_raw_arr):.1%}")

    summary = {
        "n_proteins": len(results),
        "lddt_raw_mean": float(np.mean(lddt_raw_arr)),
        "lddt_comp_mean": float(np.mean(lddt_comp_arr)),
        "lddt_retention": float(np.mean(lddt_comp_arr) / np.mean(lddt_raw_arr)),
        "contact_prec_raw_mean": float(np.mean(cp_raw_arr)),
        "contact_prec_comp_mean": float(np.mean(cp_comp_arr)),
        "contact_retention": float(np.mean(cp_comp_arr) / np.mean(cp_raw_arr)),
        "spearman_raw_mean": float(np.mean(rho_raw_arr)),
        "spearman_comp_mean": float(np.mean(rho_comp_arr)),
        "spearman_retention": float(np.mean(rho_comp_arr) / np.mean(rho_raw_arr)),
        "per_protein": results,
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 37: Structural retention benchmark")
    parser.add_argument("--n-proteins", type=int, default=50,
                        help="Number of proteins to benchmark")
    args = parser.parse_args()

    RESULTS_PATH = PROJ_ROOT / "data" / "benchmarks" / "structural_retention_results.json"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summary = run_benchmark(n_proteins=args.n_proteins)
    elapsed = time.time() - t0

    if summary:
        with open(RESULTS_PATH, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nResults saved: {RESULTS_PATH}")
        print(f"Total time: {elapsed:.1f}s")
