"""Biological annotation correlation evaluation for protein embeddings.

Maps SCOPe protein domains to biological annotations (GO terms, EC numbers,
Pfam domains, taxonomy) via SIFTS and UniProt, then evaluates how well
embedding distances correlate with annotation similarity.
"""

import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. SIFTS Mapping: SCOPe -> PDB -> UniProt
# ---------------------------------------------------------------------------


def download_sifts_mapping(
    cache_path: str = "data/annotations/sifts_mapping.json",
) -> dict[str, str]:
    """Download SIFTS PDB-chain-to-UniProt mapping and cache it.

    Downloads pdb_chain_uniprot.tsv.gz from EBI FTP.
    URL: https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz

    Returns: {pdb_chain: uniprot_accession} e.g. {"4he8_m": "P12345"}
    """
    cache = Path(cache_path)
    if cache.exists():
        print(f"  Loading cached SIFTS mapping from {cache_path}")
        with open(cache) as f:
            return json.load(f)

    cache.parent.mkdir(parents=True, exist_ok=True)

    url = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz"
    print(f"  Downloading SIFTS mapping from {url} ...")

    mapping: dict[str, str] = {}
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = gzip.decompress(resp.read()).decode("utf-8")

    for line in raw.splitlines():
        if line.startswith("#") or line.startswith("PDB"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        pdb_id = parts[0].lower()
        chain = parts[1].lower()
        uniprot = parts[2].strip()
        if uniprot:
            key = f"{pdb_id}_{chain}"
            # Keep first mapping per chain (most entries have one)
            if key not in mapping:
                mapping[key] = uniprot

    print(f"  SIFTS mapping: {len(mapping):,} PDB-chain -> UniProt entries")

    with open(cache, "w") as f:
        json.dump(mapping, f)

    return mapping


def parse_scope_to_pdb(scope_id: str) -> tuple[str, str]:
    """Parse SCOPe domain ID to PDB ID and chain.

    Example: 'd4he8m_' -> ('4he8', 'm')
    Pattern: d{pdb4}{chain}_ where chain is single letter.
    """
    # SCOPe IDs look like d4he8m_ or d1a0ha1
    # Strip leading 'd' and trailing '_' or digit
    body = scope_id.lstrip("d")
    pdb_id = body[:4].lower()
    chain = body[4].lower() if len(body) > 4 else ""
    return pdb_id, chain


def map_scope_to_uniprot(
    scope_ids: list[str],
    sifts_path: str = "data/annotations/sifts_mapping.json",
) -> dict[str, str]:
    """Map SCOPe domain IDs to UniProt accessions via SIFTS.

    Returns: {scope_id: uniprot_accession}
    Expect 60-80% coverage.
    """
    sifts = download_sifts_mapping(sifts_path)

    result: dict[str, str] = {}
    for sid in scope_ids:
        pdb_id, chain = parse_scope_to_pdb(sid)
        key = f"{pdb_id}_{chain}"
        if key in sifts:
            result[sid] = sifts[key]

    coverage = len(result) / max(len(scope_ids), 1) * 100
    print(f"  SCOPe -> UniProt mapping: {len(result)}/{len(scope_ids)} ({coverage:.1f}% coverage)")
    return result


# ---------------------------------------------------------------------------
# 2. UniProt Annotations: GO / EC / Pfam
# ---------------------------------------------------------------------------


def _uniprot_query_batch(
    accessions: list[str],
    max_retries: int = 3,
) -> dict[str, dict]:
    """Query UniProt REST API for a single batch of accessions."""
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    acc_query = " OR ".join(accessions)
    params = urllib.parse.urlencode({
        "query": f"accession:({acc_query})",
        "fields": "accession,go_id,ec,xref_pfam",
        "format": "json",
        "size": str(len(accessions)),
    })
    url = f"{base_url}?{params}"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} retries: {e}")
                return {}

    results: dict[str, dict] = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession", "")
        if not acc:
            continue

        # GO terms
        go_terms: list[str] = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "GO":
                go_id = xref.get("id", "")
                if go_id:
                    go_terms.append(go_id)

        # EC numbers
        ec_numbers: list[str] = []
        for comment in entry.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", []):
            val = comment.get("value", "")
            if val:
                ec_numbers.append(val)
        # Also check submissionNames and alternativeNames
        for name_block in entry.get("proteinDescription", {}).get("alternativeNames", []):
            for ec in name_block.get("ecNumbers", []):
                val = ec.get("value", "")
                if val and val not in ec_numbers:
                    ec_numbers.append(val)

        # Pfam domains
        pfam_domains: list[str] = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "Pfam":
                pfam_id = xref.get("id", "")
                if pfam_id:
                    pfam_domains.append(pfam_id)

        results[acc] = {
            "go": go_terms,
            "ec": ec_numbers,
            "pfam": pfam_domains,
        }

    return results


def fetch_uniprot_annotations(
    uniprot_ids: list[str],
    cache_path: str = "data/annotations/uniprot_annotations.json",
    batch_size: int = 100,
) -> dict[str, dict]:
    """Batch query UniProt REST API for GO terms, EC numbers, Pfam domains.

    Uses UniProt REST API: https://rest.uniprot.org/uniprotkb/search
    Query format: accession:(P12345 OR P67890)
    Fields: accession,go_id,ec,xref_pfam

    Returns: {uniprot_id: {"go": ["GO:0005515", ...], "ec": ["1.1.1.1", ...], "pfam": ["PF00001", ...]}}

    Caches results to avoid re-fetching.
    """
    cache = Path(cache_path)

    # Load existing cache
    cached: dict[str, dict] = {}
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)

    # Determine which IDs still need fetching
    missing = [uid for uid in uniprot_ids if uid not in cached]

    if missing:
        print(f"  Fetching annotations for {len(missing)} UniProt IDs ({len(cached)} cached)")
        cache.parent.mkdir(parents=True, exist_ok=True)

        n_batches = (len(missing) + batch_size - 1) // batch_size
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"    Batch {batch_num}/{n_batches} ({len(batch)} accessions)")

            batch_results = _uniprot_query_batch(batch)
            cached.update(batch_results)

            # Mark IDs that returned no results so we don't re-query
            for uid in batch:
                if uid not in cached:
                    cached[uid] = {"go": [], "ec": [], "pfam": []}

            # Rate limiting: 1 request per second
            if i + batch_size < len(missing):
                time.sleep(1.0)

        # Save updated cache
        with open(cache, "w") as f:
            json.dump(cached, f)
        print(f"  Cached {len(cached)} total annotations to {cache_path}")
    else:
        print(f"  All {len(uniprot_ids)} annotations loaded from cache")

    # Return only requested IDs
    return {uid: cached[uid] for uid in uniprot_ids if uid in cached}


# ---------------------------------------------------------------------------
# 3. PDB Organisms / Taxonomy
# ---------------------------------------------------------------------------


def fetch_pdb_organisms(
    pdb_ids: list[str],
    cache_path: str = "data/annotations/pdb_organisms.json",
    batch_size: int = 50,
) -> dict[str, int]:
    """Fetch source organism TaxIDs from RCSB PDB API (batch GraphQL).

    GraphQL endpoint: https://data.rcsb.org/graphql

    Returns: {pdb_id: taxid}
    """
    cache = Path(cache_path)

    cached: dict[str, int] = {}
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)
            # JSON keys are always strings; values should be ints
            cached = {k: int(v) for k, v in cached.items()}

    # Determine which IDs still need fetching
    missing = [pid for pid in pdb_ids if pid.upper() not in cached and pid.lower() not in cached]

    if missing:
        print(f"  Fetching organisms for {len(missing)} PDB IDs ({len(cached)} cached)")
        cache.parent.mkdir(parents=True, exist_ok=True)

        n_batches = (len(missing) + batch_size - 1) // batch_size
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"    Batch {batch_num}/{n_batches} ({len(batch)} PDB IDs)")

            # Build GraphQL query
            entry_ids = [pid.upper() for pid in batch]
            query = {
                "query": """
                    query($ids: [String!]!) {
                        entries(entry_ids: $ids) {
                            rcsb_id
                            polymer_entities {
                                rcsb_entity_source_organism {
                                    ncbi_taxonomy_id
                                }
                            }
                        }
                    }
                """,
                "variables": {"ids": entry_ids},
            }
            payload = json.dumps(query).encode("utf-8")

            url = "https://data.rcsb.org/graphql"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                for entry in data.get("data", {}).get("entries", []) or []:
                    rcsb_id = entry.get("rcsb_id", "").upper()
                    # Extract first organism TaxID from any polymer entity
                    taxid = None
                    for entity in entry.get("polymer_entities", []) or []:
                        for org in entity.get("rcsb_entity_source_organism", []) or []:
                            tid = org.get("ncbi_taxonomy_id")
                            if tid is not None:
                                taxid = int(tid)
                                break
                        if taxid is not None:
                            break
                    if taxid is not None:
                        cached[rcsb_id] = taxid

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                print(f"    GraphQL error: {e}")

            # Rate limiting
            if i + batch_size < len(missing):
                time.sleep(0.5)

        # Save updated cache
        with open(cache, "w") as f:
            json.dump(cached, f)
        print(f"  Cached {len(cached)} total PDB organisms to {cache_path}")
    else:
        print(f"  All {len(pdb_ids)} PDB organisms loaded from cache")

    # Return with lowercase keys for consistency with parse_scope_to_pdb
    result: dict[str, int] = {}
    for pid in pdb_ids:
        upper = pid.upper()
        lower = pid.lower()
        if upper in cached:
            result[pid] = cached[upper]
        elif lower in cached:
            result[pid] = cached[lower]

    return result


def load_ncbi_taxonomy(
    lineage_path: str = "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/TaxPointCare/poincare-embeddings/data/rankedlineage.dmp",
) -> dict[int, list[str]]:
    """Load NCBI ranked lineage file.

    Format: tax_id | tax_name | species | genus | family | order | class | phylum | kingdom | superkingdom |

    Returns: {taxid: [superkingdom, kingdom, phylum, class, order, family, genus, species]}
    """
    lineage_file = Path(lineage_path)
    if not lineage_file.exists():
        print(f"  WARNING: NCBI lineage file not found at {lineage_path}")
        return {}

    print(f"  Loading NCBI taxonomy from {lineage_path} ...")
    taxonomy: dict[int, list[str]] = {}

    with open(lineage_file) as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 10:
                continue
            try:
                taxid = int(parts[0])
            except ValueError:
                continue

            # Fields: tax_id | tax_name | species | genus | family | order | class | phylum | kingdom | superkingdom
            # We reverse to: [superkingdom, kingdom, phylum, class, order, family, genus, species]
            ranks = [
                parts[9] if len(parts) > 9 else "",  # superkingdom
                parts[8] if len(parts) > 8 else "",  # kingdom
                parts[7] if len(parts) > 7 else "",  # phylum
                parts[6] if len(parts) > 6 else "",  # class
                parts[5] if len(parts) > 5 else "",  # order
                parts[4] if len(parts) > 4 else "",  # family
                parts[3] if len(parts) > 3 else "",  # genus
                parts[2] if len(parts) > 2 else "",  # species
            ]
            taxonomy[taxid] = ranks

    print(f"  Loaded {len(taxonomy):,} taxonomy entries")
    return taxonomy


def taxonomy_distance(lineage1: list[str], lineage2: list[str]) -> int:
    """Compute taxonomy tree distance between two lineages.

    Distance = 2 * (total_ranks - shared_prefix_length)
    Higher = more distant.
    """
    total_ranks = max(len(lineage1), len(lineage2))
    if total_ranks == 0:
        return 0

    shared = 0
    for r1, r2 in zip(lineage1, lineage2):
        # Empty string means rank is unassigned — stop counting shared prefix
        if r1 and r2 and r1 == r2:
            shared += 1
        else:
            break

    return 2 * (total_ranks - shared)


# ---------------------------------------------------------------------------
# 4. Evaluation Functions
# ---------------------------------------------------------------------------


def _compute_distance_matrix(
    vectors: dict[str, np.ndarray],
    protein_ids: list[str],
    metric: str = "cosine",
) -> np.ndarray:
    """Build pairwise distance matrix for a list of protein IDs."""
    matrix = np.array([vectors[pid] for pid in protein_ids], dtype=np.float32)

    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        normed = matrix / norms
        cos_sims = normed @ normed.T
        return 1.0 - cos_sims
    else:
        return cdist(matrix, matrix, metric="euclidean").astype(np.float32)


def evaluate_go_correlation(
    vectors: dict[str, np.ndarray],
    go_terms: dict[str, list[str]],
    metric: str = "cosine",
    max_pairs: int = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate correlation between embedding distance and GO term Jaccard similarity.

    For each pair of proteins:
    - Compute GO Jaccard similarity = |GO_i & GO_j| / |GO_i | GO_j|
    - Compute embedding distance (cosine or euclidean)
    - Spearman correlation between GO similarity and embedding similarity (1 - distance)

    Returns: {spearman_rho, p_value, n_pairs, n_proteins}
    """
    rng = np.random.RandomState(seed)

    # Only include proteins with both vectors and non-empty GO terms
    pids = [pid for pid in vectors if pid in go_terms and len(go_terms[pid]) > 0]
    n = len(pids)
    if n < 5:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n_pairs": 0, "n_proteins": n}

    dist_matrix = _compute_distance_matrix(vectors, pids, metric)

    # Collect all upper-triangle pairs
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        ii, jj = np.triu_indices(n, k=1)
    else:
        # Random subsample of pairs
        ii_all, jj_all = np.triu_indices(n, k=1)
        idx = rng.choice(total_pairs, max_pairs, replace=False)
        ii = ii_all[idx]
        jj = jj_all[idx]

    # Compute GO Jaccard similarities and embedding similarities
    go_sets = {pid: set(go_terms[pid]) for pid in pids}
    jaccard_sims = np.empty(len(ii), dtype=np.float32)
    embed_sims = np.empty(len(ii), dtype=np.float32)

    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        set_i = go_sets[pids[i]]
        set_j = go_sets[pids[j]]
        union = len(set_i | set_j)
        if union > 0:
            jaccard_sims[k] = len(set_i & set_j) / union
        else:
            jaccard_sims[k] = 0.0
        embed_sims[k] = 1.0 - dist_matrix[i, j]

    rho, p_value = spearmanr(jaccard_sims, embed_sims)

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_pairs": len(ii),
        "n_proteins": n,
    }


def evaluate_ec_retrieval(
    vectors: dict[str, np.ndarray],
    ec_numbers: dict[str, list[str]],
    metric: str = "cosine",
) -> dict[str, float]:
    """Evaluate retrieval by EC number hierarchy.

    EC has 4 levels: X.X.X.X
    Evaluate Ret@1 at each level (1st digit, 2 digits, 3 digits, full).

    Returns: {ec_level1_ret1, ec_level2_ret1, ec_level3_ret1, ec_full_ret1, n_proteins}
    """
    # Only include proteins with both vectors and non-empty EC numbers
    pids = [pid for pid in vectors if pid in ec_numbers and len(ec_numbers[pid]) > 0]
    n = len(pids)
    if n < 2:
        return {
            "ec_level1_ret1": 0.0,
            "ec_level2_ret1": 0.0,
            "ec_level3_ret1": 0.0,
            "ec_full_ret1": 0.0,
            "n_proteins": n,
        }

    # Build similarity matrix
    matrix = np.array([vectors[pid] for pid in pids], dtype=np.float32)

    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        normed = matrix / norms
        sims = normed @ normed.T
    else:
        dists = cdist(matrix, matrix, metric="euclidean").astype(np.float32)
        sims = -dists  # Negate so higher = more similar

    def _ec_at_level(ec_str: str, level: int) -> str:
        """Truncate EC number to given level of hierarchy."""
        parts = ec_str.split(".")
        return ".".join(parts[:level]) if len(parts) >= level else ec_str

    # Use the first EC number per protein for simplicity
    primary_ec = {pid: ec_numbers[pid][0] for pid in pids}

    results: dict[str, float] = {}

    for level, level_name in [(1, "ec_level1_ret1"), (2, "ec_level2_ret1"),
                               (3, "ec_level3_ret1"), (4, "ec_full_ret1")]:
        # Build truncated labels
        labels = [_ec_at_level(primary_ec[pid], level) for pid in pids]

        hits = 0
        total = 0
        for i in range(n):
            row = sims[i].copy()
            row[i] = -np.inf  # Exclude self

            nn_idx = np.argmax(row)
            if labels[i] == labels[nn_idx]:
                hits += 1
            total += 1

        results[level_name] = hits / max(total, 1)

    results["n_proteins"] = n
    return results


def evaluate_pfam_retrieval(
    vectors: dict[str, np.ndarray],
    pfam_domains: dict[str, list[str]],
    metric: str = "cosine",
) -> dict[str, float]:
    """Evaluate retrieval by Pfam domain ID.

    For proteins with a single Pfam domain, evaluate if nearest neighbor
    has the same Pfam.

    Returns: {pfam_ret1, pfam_mrr, n_proteins}
    """
    # Only include proteins with exactly one Pfam domain (unambiguous label)
    pids = [pid for pid in vectors if pid in pfam_domains and len(pfam_domains[pid]) == 1]
    n = len(pids)
    if n < 2:
        return {"pfam_ret1": 0.0, "pfam_mrr": 0.0, "n_proteins": n}

    matrix = np.array([vectors[pid] for pid in pids], dtype=np.float32)
    labels = [pfam_domains[pid][0] for pid in pids]

    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        normed = matrix / norms
        sims = normed @ normed.T
    else:
        dists = cdist(matrix, matrix, metric="euclidean").astype(np.float32)
        sims = -dists

    hits = 0
    mrr_sum = 0.0

    for i in range(n):
        row = sims[i].copy()
        row[i] = -np.inf  # Exclude self

        ranked = np.argsort(row)[::-1]

        # Ret@1
        if labels[ranked[0]] == labels[i]:
            hits += 1

        # MRR
        for rank, idx in enumerate(ranked, 1):
            if labels[idx] == labels[i]:
                mrr_sum += 1.0 / rank
                break

    return {
        "pfam_ret1": hits / n,
        "pfam_mrr": mrr_sum / n,
        "n_proteins": n,
    }


def evaluate_taxonomy_correlation(
    vectors: dict[str, np.ndarray],
    taxonomy: dict[str, list[str]],
    metric: str = "cosine",
    max_pairs: int = 50_000,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate correlation between embedding distance and taxonomy tree distance.

    Returns: {spearman_rho, p_value, n_pairs, n_proteins}
    """
    rng = np.random.RandomState(seed)

    # Only include proteins with both vectors and taxonomy info
    pids = [pid for pid in vectors if pid in taxonomy and len(taxonomy[pid]) > 0]
    n = len(pids)
    if n < 5:
        return {"spearman_rho": 0.0, "p_value": 1.0, "n_pairs": 0, "n_proteins": n}

    dist_matrix = _compute_distance_matrix(vectors, pids, metric)

    # Collect all upper-triangle pairs
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii_all, jj_all = np.triu_indices(n, k=1)
        idx = rng.choice(total_pairs, max_pairs, replace=False)
        ii = ii_all[idx]
        jj = jj_all[idx]

    lineages = [taxonomy[pid] for pid in pids]

    tax_dists = np.empty(len(ii), dtype=np.float32)
    embed_sims = np.empty(len(ii), dtype=np.float32)

    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        tax_dists[k] = taxonomy_distance(lineages[i], lineages[j])
        embed_sims[k] = 1.0 - dist_matrix[i, j]

    # Taxonomy distance should anti-correlate with embedding similarity
    # (closer organisms -> higher similarity), so we correlate -tax_dist with embed_sim
    # Equivalently, correlate tax_dist with embed_dist (= 1 - embed_sim)
    rho, p_value = spearmanr(tax_dists, 1.0 - embed_sims)

    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_pairs": len(ii),
        "n_proteins": n,
    }
