"""Variant Effect Prediction (VEP) probe and ClinVar scorer for Exp 55.

Modules:
- AssayInfo / ClinVarVariant: data classes
- select_diversity_subset: pick 15 DMS assays by deterministic rules
- load_dms_assay / load_clinvar_split: ProteinGym CSV loaders
- generate_variant_sequences: WT -> per-variant mutated sequences (FASTA-ready)
- build_variant_features: per-variant feature vector for the Ridge probe
- fit_ridge_probe / score_ridge_probe: 5-fold CV with sklearn
- score_clinvar_zeroshot: cosine-distance scoring at the mutation site
- compute_assay_retention / compute_paired_retention_ci: retention math
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssayInfo:
    dms_id: str
    seq_len: int
    family: str
    fitness_type: str


@dataclass(frozen=True)
class ClinVarVariant:
    pid: str
    wt_seq: str
    mut_pos: int  # 0-indexed
    wt_aa: str
    mut_aa: str
    label: int  # 1 = pathogenic, 0 = benign


def select_diversity_subset(
    reference_df: pd.DataFrame,
    n: int = 15,
    seed: int = 42,
) -> list[AssayInfo]:
    """Pick 15 DMS assays covering 4 small / 7 medium / 4 large.

    Selection is deterministic given (df, seed). Within each size bucket we
    cycle through families and fitness types in sorted order to keep
    coverage broad before picking duplicates.
    """
    df = reference_df.copy()
    df = df[~df["DMS_id"].str.startswith("Z")]  # drop reserved Z* rows in tests

    def bucket(row):
        if row["seq_len"] < 150:
            return "small"
        if row["seq_len"] <= 400:
            return "medium"
        return "large"

    df["bucket"] = df.apply(bucket, axis=1)

    targets = {"small": 4, "medium": 7, "large": 4}
    if sum(targets.values()) != n:
        raise ValueError(f"n={n} doesn't match 4+7+4 split")

    rng = random.Random(seed)
    picked: list[AssayInfo] = []

    for buck, k in targets.items():
        sub = df[df["bucket"] == buck].sort_values("DMS_id")
        if len(sub) < k:
            raise ValueError(f"only {len(sub)} {buck} assays, need {k}")
        # Greedy: cycle through (family, fitness_type) pairs to maximize
        # diversity, then fill remainder by deterministic random sample.
        seen_pairs: set[tuple[str, str]] = set()
        chosen: list[pd.Series] = []
        for _, row in sub.iterrows():
            key = (row["family"], row["fitness_type"])
            if key not in seen_pairs and len(chosen) < k:
                chosen.append(row)
                seen_pairs.add(key)
        # Fill remainder
        if len(chosen) < k:
            remaining = [row for _, row in sub.iterrows()
                         if row["DMS_id"] not in {c["DMS_id"] for c in chosen}]
            rng.shuffle(remaining)
            chosen.extend(remaining[: k - len(chosen)])

        for row in chosen:
            picked.append(AssayInfo(
                dms_id=row["DMS_id"],
                seq_len=int(row["seq_len"]),
                family=row["family"],
                fitness_type=row["fitness_type"],
            ))
    return picked
