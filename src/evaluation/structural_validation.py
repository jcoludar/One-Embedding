"""TM-score structural validation for embedding quality assessment.

Validates that embedding similarity correlates with structural similarity
by comparing pairwise cosine similarities against pairwise TM-scores
computed from experimental PDB structures.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def extract_pdb_info(scop_domain_id: str) -> tuple[str, str]:
    """Extract PDB ID and chain from SCOPe ASTRAL domain ID.

    Format: d[pdb_4chars][chain][domain_suffix]
    Examples:
        d1dlwa_ → ('1dlw', 'A')
        d4he8m_ → ('4he8', 'M')
        d1fmta1 → ('1fmt', 'A')
    """
    pdb_id = scop_domain_id[1:5].lower()
    chain = scop_domain_id[5].upper()
    return pdb_id, chain


def fetch_pdb_structures(
    protein_ids: list[str],
    cache_dir: Path | str = Path("data/structures/pdb"),
) -> dict[str, Path]:
    """Download PDB files from RCSB for SCOPe domain IDs.

    Args:
        protein_ids: List of SCOPe domain IDs.
        cache_dir: Directory to cache downloaded PDB files.

    Returns:
        {protein_id: Path to PDB file} for successfully downloaded structures.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdb_paths = {}
    seen_pdbs = {}

    for pid in protein_ids:
        pdb_id, chain = extract_pdb_info(pid)

        # Check cache first
        pdb_file = cache_dir / f"{pdb_id}.pdb"
        if pdb_file.exists():
            pdb_paths[pid] = pdb_file
            continue

        # Download from RCSB
        if pdb_id in seen_pdbs:
            if seen_pdbs[pdb_id] is not None:
                pdb_paths[pid] = seen_pdbs[pdb_id]
            continue

        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            urllib.request.urlretrieve(url, pdb_file)
            pdb_paths[pid] = pdb_file
            seen_pdbs[pdb_id] = pdb_file
        except Exception as e:
            print(f"  Failed to download {pdb_id}: {e}")
            seen_pdbs[pdb_id] = None

    return pdb_paths


def compute_pairwise_tm_scores(
    protein_ids: list[str],
    pdb_paths: dict[str, Path],
    cache_path: Path | str | None = None,
) -> np.ndarray:
    """Compute all-pairs TM-scores using tmtools.

    Args:
        protein_ids: Ordered list of protein IDs.
        pdb_paths: {protein_id: Path to PDB file}.
        cache_path: Optional path to cache the TM-score matrix.

    Returns:
        (n, n) TM-score matrix.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = np.load(cache_path)
            return data["tm_scores"]

    try:
        import tmtools
    except ImportError:
        raise ImportError(
            "tmtools required for TM-score computation. "
            "Install with: uv pip install tmtools"
        )

    from Bio.PDB import PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    n = len(protein_ids)
    tm_scores = np.zeros((n, n), dtype=np.float32)

    # Extract CA coordinates and sequences for each protein
    ca_coords = {}
    ca_seqs = {}
    for pid in protein_ids:
        if pid not in pdb_paths:
            continue
        pdb_id, chain = extract_pdb_info(pid)
        try:
            structure = parser.get_structure(pdb_id, pdb_paths[pid])
            model = structure[0]
            if chain in model:
                chain_obj = model[chain]
            else:
                # Try first chain
                chain_obj = next(iter(model.get_chains()))

            coords = []
            seq_letters = []
            for residue in chain_obj.get_residues():
                if "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
                    resname = residue.get_resname().strip().capitalize()
                    seq_letters.append(
                        protein_letters_3to1.get(resname.lower(), "X")
                    )

            if len(coords) >= 10:
                ca_coords[pid] = np.array(coords, dtype=np.float64)
                ca_seqs[pid] = "".join(seq_letters)
        except Exception as e:
            print(f"  Failed to parse {pid}: {e}")

    # Compute pairwise TM-scores
    valid_ids = [pid for pid in protein_ids if pid in ca_coords]
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    n_errors = 0

    for i, pid_i in enumerate(valid_ids):
        for j, pid_j in enumerate(valid_ids):
            if j <= i:
                continue
            try:
                result = tmtools.tm_align(
                    ca_coords[pid_i], ca_coords[pid_j],
                    ca_seqs[pid_i], ca_seqs[pid_j],
                )
                # TM-score normalized by shorter chain
                tm = max(result.tm_norm_chain1, result.tm_norm_chain2)
                idx_i = id_to_idx[pid_i]
                idx_j = id_to_idx[pid_j]
                tm_scores[idx_i, idx_j] = tm
                tm_scores[idx_j, idx_i] = tm
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  TM-align error ({pid_i} vs {pid_j}): {e}")

    if n_errors > 0:
        print(f"  Total TM-align errors: {n_errors}")

    # Diagonal = 1.0
    np.fill_diagonal(tm_scores, 1.0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, tm_scores=tm_scores, protein_ids=protein_ids)

    return tm_scores


def validate_embedding_vs_structure(
    embedding_sim: np.ndarray,
    tm_scores: np.ndarray,
    protein_ids: list[str],
    metadata: list[dict] | None = None,
) -> dict:
    """Correlate embedding similarity with structural TM-scores.

    Args:
        embedding_sim: (n, n) pairwise cosine similarity matrix.
        tm_scores: (n, n) pairwise TM-score matrix.
        protein_ids: Ordered list of protein IDs.
        metadata: Optional metadata for per-class breakdown.

    Returns:
        Dict with Spearman ρ, Pearson r, and per-class breakdown.
    """
    n = len(protein_ids)

    # Extract upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(n, k=1)
    emb_flat = embedding_sim[triu_idx]
    tm_flat = tm_scores[triu_idx]

    # Filter out pairs with zero TM-score (missing structures)
    valid = tm_flat > 0
    emb_valid = emb_flat[valid]
    tm_valid = tm_flat[valid]

    if len(emb_valid) < 10:
        return {"error": "Too few valid pairs", "n_valid_pairs": int(valid.sum())}

    spearman_rho, spearman_p = spearmanr(emb_valid, tm_valid)
    pearson_r, pearson_p = pearsonr(emb_valid, tm_valid)

    result = {
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_proteins": n,
        "n_valid_pairs": int(valid.sum()),
        "n_total_pairs": int(len(emb_flat)),
    }

    # Per-class breakdown
    if metadata is not None:
        id_to_class = {m["id"]: m.get("class_name", "") for m in metadata}
        classes = sorted(set(id_to_class.values()) - {""})

        per_class = {}
        for cls in classes:
            cls_mask = np.array(
                [id_to_class.get(pid, "") == cls for pid in protein_ids]
            )
            cls_idx = np.where(cls_mask)[0]
            if len(cls_idx) < 5:
                continue

            # Get pairs within this class
            pairs = []
            for i in range(len(cls_idx)):
                for j in range(i + 1, len(cls_idx)):
                    ii, jj = cls_idx[i], cls_idx[j]
                    if tm_scores[ii, jj] > 0:
                        pairs.append((embedding_sim[ii, jj], tm_scores[ii, jj]))

            if len(pairs) >= 10:
                e, t = zip(*pairs)
                rho, p = spearmanr(e, t)
                per_class[cls] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "n_pairs": len(pairs),
                }

        result["per_class"] = per_class

    return result
