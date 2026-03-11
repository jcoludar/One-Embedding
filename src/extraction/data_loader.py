"""FASTA I/O, protein set curation, and SCOPe data handling."""

import csv
import re
from pathlib import Path

from Bio import SeqIO


def read_fasta(fasta_path: Path | str) -> dict[str, str]:
    """Read FASTA file into {id: sequence} dict."""
    records = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        records[record.id] = str(record.seq)
    return records


def write_fasta(records: dict[str, str], fasta_path: Path | str) -> None:
    """Write {id: sequence} dict to FASTA."""
    fasta_path = Path(fasta_path)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as f:
        for sid, seq in records.items():
            f.write(f">{sid}\n{seq}\n")


def parse_scope_header(header: str) -> dict:
    """Parse SCOPe ASTRAL FASTA header.

    Example header:
    d1dlwa_ a.1.1.1 (A:) Myoglobin {Sperm whale ...}

    Returns dict with keys: sid, sccs, family, superfamily, fold, class_name.
    """
    parts = header.split()
    sid = parts[0]
    sccs = parts[1] if len(parts) > 1 else ""

    # Parse SCCS (e.g., a.1.1.1)
    sccs_parts = sccs.split(".")
    class_name = sccs_parts[0] if len(sccs_parts) > 0 else ""
    fold = ".".join(sccs_parts[:2]) if len(sccs_parts) >= 2 else ""
    superfamily = ".".join(sccs_parts[:3]) if len(sccs_parts) >= 3 else ""
    family = sccs if len(sccs_parts) >= 4 else ""

    return {
        "sid": sid,
        "sccs": sccs,
        "family": family,
        "superfamily": superfamily,
        "fold": fold,
        "class_name": class_name,
    }


CLASS_LABELS = {
    "a": "all-alpha",
    "b": "all-beta",
    "c": "alpha/beta",
    "d": "alpha+beta",
    "e": "multi-domain",
    "f": "membrane",
    "g": "small",
}


def curate_scope_set(
    scope_fasta: Path | str,
    n_proteins: int = 100,
    min_length: int = 50,
    max_length: int = 500,
    n_per_family: int = 5,
    seed: int = 42,
) -> tuple[dict[str, str], list[dict]]:
    """Curate a diverse subset of SCOPe proteins.

    Returns (fasta_dict, metadata_rows) where metadata_rows is a list of dicts
    with keys: id, family, superfamily, fold, class_name, class_label, length.
    """
    import random

    random.seed(seed)

    # Parse all sequences with metadata
    all_proteins = []
    for record in SeqIO.parse(str(scope_fasta), "fasta"):
        seq = str(record.seq)
        # Filter by length and valid amino acids
        if len(seq) < min_length or len(seq) > max_length:
            continue
        if re.search(r"[^ACDEFGHIKLMNPQRSTVWY]", seq.upper()):
            continue

        meta = parse_scope_header(record.description)
        if not meta["family"]:
            continue

        all_proteins.append({
            "id": record.id,
            "sequence": seq.upper(),
            **meta,
        })

    # Group by family
    families: dict[str, list] = {}
    for p in all_proteins:
        families.setdefault(p["family"], []).append(p)

    # Select families, preferring multi-member ones, stratified by class
    selected = []
    multi_families_by_class: dict[str, list[str]] = {}
    single_families_by_class: dict[str, list[str]] = {}
    for fam, members in families.items():
        cls = members[0]["class_name"]
        if len(members) >= 2:
            multi_families_by_class.setdefault(cls, []).append(fam)
        else:
            single_families_by_class.setdefault(cls, []).append(fam)

    # First, fill from multi-member families (better for retrieval eval)
    all_classes = sorted(set(list(multi_families_by_class.keys()) + list(single_families_by_class.keys())))
    n_classes = len(all_classes)
    if n_classes == 0:
        return {}, []

    target_per_class = max(1, n_proteins // n_classes)

    for cls in all_classes:
        count = 0
        # Multi-member families first
        multi_fams = multi_families_by_class.get(cls, [])
        random.shuffle(multi_fams)
        for fam in multi_fams:
            if count >= target_per_class:
                break
            members = families[fam]
            random.shuffle(members)
            take = min(n_per_family, len(members), target_per_class - count)
            selected.extend(members[:take])
            count += take

        # Fill remaining with singletons
        single_fams = single_families_by_class.get(cls, [])
        random.shuffle(single_fams)
        for fam in single_fams:
            if count >= target_per_class:
                break
            selected.extend(families[fam][:1])
            count += 1

    # Trim to target size
    random.shuffle(selected)
    selected = selected[:n_proteins]

    fasta_dict = {p["id"]: p["sequence"] for p in selected}
    metadata = [
        {
            "id": p["id"],
            "family": p["family"],
            "superfamily": p["superfamily"],
            "fold": p["fold"],
            "class_name": p["class_name"],
            "class_label": CLASS_LABELS.get(p["class_name"], p["class_name"]),
            "length": len(p["sequence"]),
        }
        for p in selected
    ]

    return fasta_dict, metadata


def filter_by_family_size(
    metadata: list[dict],
    min_members: int = 3,
    label_key: str = "family",
) -> tuple[list[dict], set[str]]:
    """Filter to proteins in families with >= min_members.

    Returns (filtered_metadata, kept_ids).
    """
    from collections import Counter

    counts = Counter(m[label_key] for m in metadata)
    kept = [m for m in metadata if counts[m[label_key]] >= min_members]
    return kept, {m["id"] for m in kept}


def save_metadata_csv(metadata: list[dict], csv_path: Path | str) -> None:
    """Save metadata list of dicts to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata:
        return
    fieldnames = list(metadata[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


def load_metadata_csv(csv_path: Path | str) -> list[dict]:
    """Load metadata CSV into list of dicts."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))
