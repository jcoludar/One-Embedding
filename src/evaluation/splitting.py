"""Superfamily-aware train/test splitting for protein evaluation.

Ensures no superfamily leakage: entire superfamilies go to one side of the split,
so structurally homologous proteins never appear in both train and test.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def superfamily_aware_split(
    metadata: list[dict],
    test_fraction: float = 0.3,
    min_test_family_size: int = 2,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split protein IDs ensuring no superfamily leakage.

    Entire superfamilies are assigned to train or test (no straddling).
    Stratified by structural class (a/b/c/d/e/f/g) for balance.

    Args:
        metadata: List of dicts with 'id', 'superfamily', 'family', 'class_name'.
        test_fraction: Fraction of superfamilies to assign to test.
        min_test_family_size: Minimum family members in test for retrieval eval.
        seed: Random seed for reproducibility.

    Returns:
        (train_ids, test_ids, eval_ids) where eval_ids is a subset of test_ids
        restricted to proteins whose family has >= min_test_family_size members
        in the test set.
    """
    rng = random.Random(seed)

    # Group superfamilies by structural class
    sf_to_class: dict[str, str] = {}
    sf_to_ids: dict[str, list[str]] = defaultdict(list)
    for m in metadata:
        sf = m["superfamily"]
        sf_to_class[sf] = m["class_name"]
        sf_to_ids[sf].append(m["id"])

    class_to_sfs: dict[str, list[str]] = defaultdict(list)
    for sf, cls in sf_to_class.items():
        class_to_sfs[cls].append(sf)

    # Stratified split: within each class, assign superfamilies to test
    test_sfs: set[str] = set()
    for cls in sorted(class_to_sfs.keys()):
        sfs = sorted(class_to_sfs[cls])  # sort for determinism
        rng.shuffle(sfs)
        n_test = max(1, round(len(sfs) * test_fraction))
        test_sfs.update(sfs[:n_test])

    # Assign proteins
    train_ids = []
    test_ids = []
    for m in metadata:
        if m["superfamily"] in test_sfs:
            test_ids.append(m["id"])
        else:
            train_ids.append(m["id"])

    # Compute eval_ids: test proteins whose family has enough test-set members
    test_id_set = set(test_ids)
    family_counts_in_test = Counter(
        m["family"] for m in metadata if m["id"] in test_id_set
    )
    eval_ids = [
        m["id"]
        for m in metadata
        if m["id"] in test_id_set
        and family_counts_in_test[m["family"]] >= min_test_family_size
    ]

    return train_ids, test_ids, eval_ids


def split_statistics(
    metadata: list[dict],
    train_ids: list[str],
    test_ids: list[str],
    eval_ids: list[str],
) -> dict:
    """Compute and return split statistics for logging."""
    train_set = set(train_ids)
    test_set = set(test_ids)

    train_meta = [m for m in metadata if m["id"] in train_set]
    test_meta = [m for m in metadata if m["id"] in test_set]

    train_sfs = set(m["superfamily"] for m in train_meta)
    test_sfs = set(m["superfamily"] for m in test_meta)
    overlap = train_sfs & test_sfs

    train_fams = set(m["family"] for m in train_meta)
    test_fams = set(m["family"] for m in test_meta)

    return {
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "n_eval": len(eval_ids),
        "n_train_superfamilies": len(train_sfs),
        "n_test_superfamilies": len(test_sfs),
        "superfamily_overlap": len(overlap),
        "n_train_families": len(train_fams),
        "n_test_families": len(test_fams),
        "family_overlap": len(train_fams & test_fams),
    }


def save_split(
    train_ids: list[str],
    test_ids: list[str],
    eval_ids: list[str],
    path: Path | str,
    stats: dict | None = None,
) -> None:
    """Save split to JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "eval_ids": eval_ids,
    }
    if stats:
        data["statistics"] = stats
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_split(path: Path | str) -> tuple[list[str], list[str], list[str]]:
    """Load split from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["train_ids"], data["test_ids"], data["eval_ids"]


def family_stratified_split(
    metadata: list[dict],
    test_fraction: float = 0.3,
    min_family_size: int = 2,
    label_key: str = "family",
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split proteins so every family (with enough members) appears in both train and test.

    For families with >= min_family_size members, proteins are split within
    each family so both sides have representatives. Families with fewer members
    go entirely to train.

    This split is appropriate for classification evaluation where the classifier
    needs to see each label during training.

    Args:
        metadata: List of dicts with 'id' and label_key.
        test_fraction: Fraction of each family's members to assign to test.
        min_family_size: Minimum family size to split; smaller families go to train only.
        label_key: Which label column defines families.
        seed: Random seed for reproducibility.

    Returns:
        (cls_train_ids, cls_test_ids)
    """
    rng = random.Random(seed)

    # Group proteins by family
    family_to_ids: dict[str, list[str]] = defaultdict(list)
    for m in metadata:
        family_to_ids[m[label_key]].append(m["id"])

    cls_train_ids = []
    cls_test_ids = []

    for fam in sorted(family_to_ids.keys()):
        members = sorted(family_to_ids[fam])  # sort for determinism
        rng.shuffle(members)

        if len(members) < min_family_size:
            # Too small to split — train only
            cls_train_ids.extend(members)
        else:
            n_test = max(1, round(len(members) * test_fraction))
            cls_test_ids.extend(members[:n_test])
            cls_train_ids.extend(members[n_test:])

    return cls_train_ids, cls_test_ids


def split_embeddings(
    embeddings: dict,
    train_ids: list[str],
    test_ids: list[str],
) -> tuple[dict, dict]:
    """Partition embeddings dict into train and test subsets."""
    train_set = set(train_ids)
    test_set = set(test_ids)
    train_emb = {k: v for k, v in embeddings.items() if k in train_set}
    test_emb = {k: v for k, v in embeddings.items() if k in test_set}
    return train_emb, test_emb
