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


def prepare_reference_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename real ProteinGym DMS_substitutions.csv columns to canonical names.

    Maps:
        taxon                  -> family
        coarse_selection_type  -> fitness_type

    Pass-through if the canonical names are already present (so synthetic
    test data with ``family`` / ``fitness_type`` columns still works without
    going through this helper).

    Raises ValueError if neither the canonical nor the source column is
    present, listing what was missing.
    """
    out = df.copy()

    if "family" not in out.columns:
        if "taxon" not in out.columns:
            raise ValueError(
                "reference df must have either 'family' or 'taxon' column"
            )
        out["family"] = out["taxon"]

    if "fitness_type" not in out.columns:
        if "coarse_selection_type" not in out.columns:
            raise ValueError(
                "reference df must have either 'fitness_type' or "
                "'coarse_selection_type' column"
            )
        out["fitness_type"] = out["coarse_selection_type"]

    if "seq_len" not in out.columns:
        if "target_seq" in out.columns:
            out["seq_len"] = out["target_seq"].str.len()
        else:
            raise ValueError(
                "reference df must have either 'seq_len' or 'target_seq'"
            )

    if "DMS_id" not in out.columns:
        raise ValueError("reference df must have 'DMS_id' column")

    return out


# ---------------------------------------------------------------------------
# ProteinGym CSV loaders (Task 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMSVariant:
    mut_pos: int  # 0-indexed
    wt_aa: str
    mut_aa: str
    score: float
    mutated_sequence: str


@dataclass(frozen=True)
class DMSAssay:
    dms_id: str
    wt_sequence: str
    variants: list[DMSVariant]


def _parse_mutant(mut_str: str) -> tuple[int, str, str]:
    """'M1A' -> (0, 'M', 'A'). One-indexed position in input -> 0-indexed."""
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos_1idx = int(mut_str[1:-1])
    if pos_1idx < 1:
        raise ValueError(
            f"_parse_mutant: position must be >= 1 (got {pos_1idx!r} from {mut_str!r})"
        )
    return pos_1idx - 1, wt_aa, mut_aa


def load_dms_assay(csv_path: str, dms_id: str) -> DMSAssay:
    """Load one ProteinGym DMS CSV. Recovers WT sequence by inverting any variant."""
    df = pd.read_csv(csv_path)
    needed = {"mutant", "mutated_sequence", "DMS_score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{csv_path} missing columns; have {list(df.columns)}")
    if df.empty:
        raise ValueError(f"{csv_path} has no data rows")

    # Recover WT by inverting the first variant.
    first = df.iloc[0]
    pos_0, wt_aa, _mut_aa = _parse_mutant(first["mutant"])
    mut_seq = first["mutated_sequence"]
    wt_sequence = mut_seq[:pos_0] + wt_aa + mut_seq[pos_0 + 1:]

    variants: list[DMSVariant] = []
    for _, row in df.iterrows():
        pos, wt, mut = _parse_mutant(row["mutant"])
        variants.append(DMSVariant(
            mut_pos=pos,
            wt_aa=wt,
            mut_aa=mut,
            score=float(row["DMS_score"]),
            mutated_sequence=str(row["mutated_sequence"]),
        ))
    return DMSAssay(
        dms_id=dms_id,
        wt_sequence=wt_sequence,
        variants=variants,
    )


def load_clinvar_split(csv_path: str, pid: str) -> list[ClinVarVariant]:
    """Load one ProteinGym clinical CSV (per parent protein)."""
    df = pd.read_csv(csv_path)
    needed = {"mutant", "mutated_sequence", "DMS_bin_score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{csv_path} missing columns; have {list(df.columns)}")
    if df.empty:
        raise ValueError(f"{csv_path} has no data rows")

    first = df.iloc[0]
    pos_0, wt_aa, _ = _parse_mutant(first["mutant"])
    mut_seq = first["mutated_sequence"]
    wt_sequence = mut_seq[:pos_0] + wt_aa + mut_seq[pos_0 + 1:]

    out: list[ClinVarVariant] = []
    for _, row in df.iterrows():
        pos, wt, mut = _parse_mutant(row["mutant"])
        out.append(ClinVarVariant(
            pid=pid,
            wt_seq=wt_sequence,
            mut_pos=pos,
            wt_aa=wt,
            mut_aa=mut,
            label=int(row["DMS_bin_score"]),
        ))
    return out


# ---------------------------------------------------------------------------
# Task 4: variant feature builder
# ---------------------------------------------------------------------------

def build_variant_features(
    wt_emb: np.ndarray,
    mut_emb: np.ndarray,
    mut_pos: int,
) -> np.ndarray:
    """Build the 4*D feature vector for one variant.

    Args:
        wt_emb: (L, D) per-residue WT embedding.
        mut_emb: (L, D) per-residue mutant embedding (same L, same D).
        mut_pos: 0-indexed mutation position.

    Returns:
        (4*D,) feature: [wt_emb[mut_pos], mut_emb[mut_pos], mean(wt_emb),
        mean(mut_emb)].
    """
    if wt_emb.shape != mut_emb.shape:
        raise ValueError(f"shape mismatch: {wt_emb.shape} vs {mut_emb.shape}")
    if mut_pos < 0 or mut_pos >= wt_emb.shape[0]:
        raise IndexError(f"mut_pos {mut_pos} out of range for L={wt_emb.shape[0]}")
    parts = [
        wt_emb[mut_pos],
        mut_emb[mut_pos],
        wt_emb.mean(axis=0),
        mut_emb.mean(axis=0),
    ]
    return np.concatenate(parts).astype(np.float32)
