"""CATH-level cluster splits for Seq2OE experiments.

Loads the CATH20 labeled FASTA (headers of the form `>{pid}|{C}.{A}.{T}.{H}`)
and produces whole-cluster holdout splits at the Homologous-Superfamily (H)
or Topology (T) level with per-Class greedy stratification.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def parse_cath_fasta(path: Path | str) -> dict[str, dict]:
    """Parse a CATH-labeled FASTA.

    Expected header format: `>{pid}|{C}.{A}.{T}.{H}` where C/A/T/H are CATH
    Class / Architecture / Topology / Homologous-Superfamily codes, e.g.
    `>12asA00|3.30.930.10`.

    Returns a dict mapping protein id to:
        {
            "seq": str,
            "C": int,          # class integer
            "A": int,          # architecture integer
            "T": str,          # topology dotted code (e.g. "3.30.930")
            "H": str,          # homologous-superfamily dotted code (full)
        }

    Raises ValueError on malformed headers or codes.
    """
    path = Path(path)
    meta: dict[str, dict] = {}
    current_id: str | None = None
    current_info: dict | None = None
    seq_lines: list[str] = []

    def flush():
        if current_id is not None:
            current_info["seq"] = "".join(seq_lines)
            meta[current_id] = current_info

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                if "|" not in header:
                    raise ValueError(f"Header has no CATH code: {header!r}")
                pid, code = header.split("|", 1)
                parts = code.split(".")
                if len(parts) != 4:
                    raise ValueError(
                        f"CATH code {code!r} expected 4 dot-separated fields"
                    )
                try:
                    c_int = int(parts[0])
                    a_int = int(parts[1])
                except ValueError as e:
                    raise ValueError(f"Non-integer C/A in {code!r}") from e
                current_id = pid
                current_info = {
                    "C": c_int,
                    "A": a_int,
                    "T": ".".join(parts[:3]),
                    "H": code,
                }
                seq_lines = []
            else:
                seq_lines.append(line)
        flush()

    return meta


def cath_cluster_split(
    metadata: dict[str, dict],
    level: str,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Whole-cluster holdout split of CATH-labeled proteins.

    Groups proteins by the chosen CATH level (H or T), then assigns whole
    groups to train / val / test folds. Within each Class (C), groups are
    shuffled deterministically from the seed and walked in order; each group
    goes to whichever fold is currently furthest below its target fraction
    (measured in number of proteins, not groups). This per-Class greedy
    strategy keeps every class proportionally represented in every fold.

    Args:
        metadata: Output of `parse_cath_fasta` — dict of pid -> info dict with
            keys "C", "T", "H" (among others).
        level: "H" (homologous superfamily) or "T" (topology/fold).
        fractions: (train, val, test) fractions. Must sum to 1.0.
        seed: RNG seed controlling shuffle order.

    Returns:
        (train_ids, val_ids, test_ids), each sorted alphabetically.
    """
    if level not in ("H", "T"):
        raise ValueError(f"level must be 'H' or 'T', got {level!r}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    # Group proteins by (class, cluster code)
    class_to_groups: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pid, info in metadata.items():
        cls = info["C"]
        cluster = info[level]
        class_to_groups[cls][cluster].append(pid)

    folds: list[list[str]] = [[], [], []]
    rng = random.Random(seed)

    for cls in sorted(class_to_groups.keys()):
        groups = class_to_groups[cls]
        # Deterministic shuffle of group codes (sort first so the shuffle is
        # immune to dict insertion order, then shuffle with the seeded RNG)
        group_codes = sorted(groups.keys())
        rng.shuffle(group_codes)

        # Total proteins in this class
        class_total = sum(len(groups[g]) for g in group_codes)
        targets = [f * class_total for f in fractions]
        class_counts = [0, 0, 0]

        for g in group_codes:
            members = sorted(groups[g])
            # Choose the fold furthest below its target population.
            # Tie-break: list.index returns the lowest matching index, so
            # equal deficits bias toward train, then val. Deterministic
            # across Python versions.
            deficits = [t - c for t, c in zip(targets, class_counts)]
            chosen = deficits.index(max(deficits))
            folds[chosen].extend(members)
            class_counts[chosen] += len(members)

    train = sorted(folds[0])
    val = sorted(folds[1])
    test = sorted(folds[2])
    return train, val, test


def save_split(
    path: Path | str,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    meta: dict,
) -> None:
    """Save a split + its metadata to JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
        "meta": meta,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(
    path: Path | str,
) -> tuple[list[str], list[str], list[str], dict]:
    """Load a split saved by `save_split`."""
    with open(path) as f:
        payload = json.load(f)
    return (
        payload["train_ids"],
        payload["val_ids"],
        payload["test_ids"],
        payload["meta"],
    )
