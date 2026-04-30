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


# ---------------------------------------------------------------------------
# Task 5: Ridge probe with 5-fold CV and multi-seed averaging
# ---------------------------------------------------------------------------

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass(frozen=True)
class ProbeResult:
    spearman_rho: float
    predictions: np.ndarray  # (n_variants,) — averaged over seeds, OOF
    n_variants: int


def fit_evaluate_ridge_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seeds: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    inner_cv: int = 3,
) -> ProbeResult:
    """5-fold CV Ridge probe with inner GridSearchCV on alpha; multi-seed averaging.

    Per outer fold: pick best alpha by inner CV on train, predict on test.
    Predictions aggregated OOF across all folds. Repeated for each seed
    (seed controls outer-fold split + inner-CV split). Final OOF predictions
    are the mean across seeds. Reports Spearman ρ on the averaged predictions.

    Args:
        X: (n_variants, feat_dim) feature matrix.
        y: (n_variants,) target values.
        n_folds: number of outer CV folds (default 5).
        seeds: list of integer seeds for KFold shuffles; averaged over seeds.
            Defaults to [42, 123, 456].
        alpha_grid: list of Ridge alpha values to sweep in inner CV.
            Defaults to ALPHA_GRID.
        inner_cv: number of inner CV folds for alpha selection (default 3).

    Returns:
        ProbeResult with Spearman ρ, averaged OOF predictions, and n_variants.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    from scipy.stats import spearmanr

    if seeds is None:
        seeds = [42, 123, 456]
    if alpha_grid is None:
        alpha_grid = ALPHA_GRID

    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = len(y)

    seed_preds: list[np.ndarray] = []
    for seed in seeds:
        oof = np.zeros(n, dtype=np.float64)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            inner = GridSearchCV(
                estimator=Ridge(),
                param_grid={"alpha": alpha_grid},
                cv=inner_cv,
                scoring="r2",
                n_jobs=1,
            )
            inner.fit(X[tr], y[tr])
            oof[te] = inner.best_estimator_.predict(X[te])
        seed_preds.append(oof)

    avg_preds = np.mean(seed_preds, axis=0).astype(np.float32)
    rho_val, _ = spearmanr(y, avg_preds)
    return ProbeResult(
        spearman_rho=float(rho_val),
        predictions=avg_preds,
        n_variants=n,
    )


# ---------------------------------------------------------------------------
# Task 6: ClinVar zero-shot scorer
# ---------------------------------------------------------------------------

def score_clinvar_zeroshot(
    wt_emb: np.ndarray,
    mut_emb: np.ndarray,
    mut_pos: int,
) -> float:
    """Cosine-distance score at the mutation site. Returns 1 - cos(WT[p], mut[p]).

    A score of 0 means identical embeddings (no predicted effect).
    A score of 1 means orthogonal embeddings (maximum predicted effect).
    When either embedding vector has zero norm (undefined cosine), returns 1.0
    as the conservative worst-case estimate.

    Args:
        wt_emb: (L, D) per-residue wild-type embedding.
        mut_emb: (L, D) per-residue mutant embedding.
        mut_pos: 0-indexed mutation position.

    Returns:
        Float in [0, 2] (typically [0, 1] for non-antiparallel vectors).

    Raises:
        IndexError: if mut_pos is out of bounds.
    """
    if mut_pos < 0 or mut_pos >= wt_emb.shape[0]:
        raise IndexError(f"mut_pos {mut_pos} out of range for L={wt_emb.shape[0]}")
    a = wt_emb[mut_pos]
    b = mut_emb[mut_pos]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0  # undefined cosine — worst case
    cos = float(np.dot(a, b) / (na * nb))
    return 1.0 - cos


def clinvar_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC for ClinVar binary labels (1 = pathogenic).

    Args:
        scores: (n,) float array of pathogenicity scores (higher = more pathogenic).
        labels: (n,) int array with 1 = pathogenic, 0 = benign.

    Returns:
        ROC AUC as a float in [0, 1].
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Task 7: BCa bootstrap CI helpers
# ---------------------------------------------------------------------------

def _bca_ci(samples: np.ndarray, point: float, alpha: float = 0.05) -> tuple[float, float]:
    """BCa CI helper — uses scipy if available, else percentile fallback."""
    from scipy.stats import norm

    samples = np.sort(samples)
    n = len(samples)
    # Bias correction
    p_below = float(np.mean(samples < point))
    if p_below in (0.0, 1.0):
        # Fall back to percentile interval
        lo = float(np.quantile(samples, alpha / 2))
        hi = float(np.quantile(samples, 1 - alpha / 2))
        return lo, hi
    z0 = norm.ppf(p_below)
    # Acceleration via jackknife on the samples array
    jack = np.array([np.delete(samples, i).mean() for i in range(min(n, 200))])
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6.0 * (((jack_mean - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)
    p_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    p_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    lo = float(np.quantile(samples, p_lo))
    hi = float(np.quantile(samples, p_hi))
    return lo, hi


def bootstrap_ci_paired(
    raw_per_assay: np.ndarray,
    codec_per_assay: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI on retention = mean(codec) / mean(raw) × 100.

    Uses paired resampling (same indices for both arrays each bootstrap
    iteration) and the ratio-of-means statistic. Matches the protocol used
    in Exp 43/44/47 (see experiments/43_rigorous_benchmark/metrics/
    statistics.py::paired_bootstrap_retention).

    Args:
        raw_per_assay: (n,) per-assay scores under the raw (uncompressed)
            condition. n_assays should be >= 5 for a meaningful CI.
        codec_per_assay: (n,) per-assay scores under the compressed condition.
        n_boot: number of bootstrap resamples (default 10,000).
        seed: random seed.

    Returns:
        dict with keys retention_pct, ci_low, ci_high, n_assays.
    """
    raw = np.asarray(raw_per_assay, dtype=np.float64)
    codec = np.asarray(codec_per_assay, dtype=np.float64)
    if raw.shape != codec.shape:
        raise ValueError("paired arrays must match")
    n = len(raw)
    rng = np.random.default_rng(seed)
    if raw.mean() == 0:
        # Degenerate: raw is all zeros. Return a "no signal" result.
        return {
            "retention_pct": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_assays": n,
        }
    point = 100.0 * codec.mean() / raw.mean()
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rm = raw[idx].mean()
        if rm == 0:
            samples[b] = 0.0
        else:
            samples[b] = 100.0 * codec[idx].mean() / rm
    lo, hi = _bca_ci(samples, point)
    return {
        "retention_pct": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_assays": n,
    }


def bootstrap_ci_pearson(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap Pearson r with BCa CIs for RNS↔VEP correlation.

    Args:
        x: (n,) first variable array.
        y: (n,) second variable array.
        n_boot: number of bootstrap resamples (default 10,000).
        seed: random seed for reproducibility.

    Returns:
        dict with keys:
            pearson_r: point estimate.
            ci_low: lower 95% BCa CI bound.
            ci_high: upper 95% BCa CI bound.
            n: sample size.
    """
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x, y must match")
    n = len(x)
    rng = np.random.default_rng(seed)
    point, _ = pearsonr(x, y)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            r, _ = pearsonr(x[idx], y[idx])
        except Exception:
            r = 0.0
        samples[b] = r
    lo, hi = _bca_ci(samples, float(point))
    return {
        "pearson_r": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
    }
