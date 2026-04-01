# Disorder Information Loss Investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exploratory notebook that identifies exactly where and why the codec loses disorder prediction signal, then convert findings into a reproducible script.

**Architecture:** A single notebook (`experiments/45_disorder_forensics.ipynb`) structured as 5 layers. Each layer is a self-contained section that loads results from the previous layer. A shared helper module (`experiments/45_disorder_helpers.py`) holds reusable functions for both the notebook and the eventual validation script.

**Tech Stack:** numpy, scipy, h5py, matplotlib, sklearn (Ridge), seaborn for heatmaps. No new dependencies.

---

### Task 1: Helper Module — Data Loading and Stage Transforms

**Files:**
- Create: `experiments/45_disorder_helpers.py`

This module provides the core functions used by both the notebook and future validation script. Each function handles one pipeline stage in isolation.

- [ ] **Step 1: Create helper module with loading and transform functions**

```python
# experiments/45_disorder_helpers.py
"""Shared helpers for disorder information loss investigation (Exp 45).

Functions for loading data, applying individual codec stages, training
per-stage Ridge probes, and computing per-protein Spearman rho.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.one_embedding.preprocessing import (
    compute_corpus_stats,
    center_embeddings,
    all_but_the_top,
)
from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.evaluation.per_residue_tasks import load_chezod_seth

DATA = ROOT / "data"


# ── Data loading ──────────────────────────────────────────────────────

def load_h5_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load {protein_id: (L, D) float32} from flat H5."""
    embs = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            embs[key] = f[key][:].astype(np.float32)
    return embs


def load_chezod_data():
    """Load CheZOD sequences, Z-scores, and train/test split."""
    data_dir = DATA / "per_residue_benchmarks"
    sequences, scores, train_ids, test_ids = load_chezod_seth(data_dir)
    embs = load_h5_embeddings(DATA / "residue_embeddings" / "prot_t5_xl_chezod.h5")
    # Filter to proteins with both embeddings and scores
    train_ids = [p for p in train_ids if p in embs and p in scores]
    test_ids = [p for p in test_ids if p in embs and p in scores]
    return embs, scores, train_ids, test_ids


def load_trizod_data():
    """Load TriZOD embeddings, Z-scores, and train/test split."""
    import json
    split_path = DATA / "benchmark_suite" / "splits" / "trizod_predefined.json"
    with open(split_path) as f:
        split = json.load(f)
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]

    embs = load_h5_embeddings(DATA / "residue_embeddings" / "prot_t5_xl_trizod.h5")

    # Load TriZOD scores
    sys.path.insert(0, str(ROOT / "experiments" / "43_rigorous_benchmark"))
    from datasets.trizod import load_trizod_embeddings
    _, scores = load_trizod_embeddings()

    train_ids = [p for p in train_ids if p in embs and p in scores]
    test_ids = [p for p in test_ids if p in embs and p in scores]
    return embs, scores, train_ids, test_ids


# ── Pipeline stages (each applied independently) ─────────────────────

def apply_abtt(embs: dict, corpus_stats: dict, k: int = 3) -> dict:
    """Apply centering + top-k PC removal to all proteins."""
    mean_vec = corpus_stats["mean_vec"]
    top_pcs = corpus_stats["top_pcs"][:k]
    result = {}
    for pid, emb in embs.items():
        centered = center_embeddings(emb, mean_vec)
        result[pid] = all_but_the_top(centered, top_pcs)
    return result


def apply_rp(embs: dict, d_out: int = 768, seed: int = 42) -> dict:
    """Apply random orthogonal projection to d_out dimensions."""
    # Build projection matrix (same as codec)
    sample_d = next(iter(embs.values())).shape[1]
    rng = np.random.RandomState(seed)
    R = rng.randn(sample_d, d_out).astype(np.float32)
    Q, _ = np.linalg.qr(R, mode="reduced")
    proj = Q * np.sqrt(sample_d / d_out)
    return {pid: (emb @ proj) for pid, emb in embs.items()}


def apply_pq(embs: dict, codec: OneEmbeddingCodec) -> dict:
    """Apply PQ encode + decode using a fitted codec's PQ model."""
    from src.one_embedding.quantization import pq_encode, pq_decode
    result = {}
    for pid, emb in embs.items():
        codes = pq_encode(emb, codec._pq_model)
        result[pid] = pq_decode(codes, codec._pq_model)
    return result


def apply_fp16(embs: dict) -> dict:
    """Apply fp16 quantization (cast down and back)."""
    return {pid: emb.astype(np.float16).astype(np.float32) for pid, emb in embs.items()}


# ── Ridge probe training + per-protein evaluation ────────────────────

def train_ridge_probe(embs: dict, scores: dict, train_ids: list,
                      max_len: int = 512,
                      alpha_grid: list = None):
    """Train RidgeCV on stacked residue embeddings, return fitted model."""
    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    X_parts, y_parts = [], []
    for pid in train_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))
        for i in range(n):
            if not np.isnan(lab[i]):
                X_parts.append(emb[i])
                y_parts.append(lab[i])

    X = np.stack(X_parts).astype(np.float32)
    y = np.array(y_parts, dtype=np.float64)
    model = RidgeCV(alphas=alpha_grid, cv=3)
    model.fit(X, y)
    return model


def per_protein_rho(embs: dict, scores: dict, protein_ids: list,
                    probe, max_len: int = 512,
                    min_residues: int = 5) -> dict[str, float]:
    """Compute per-protein Spearman rho using a fitted probe."""
    results = {}
    for pid in protein_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))

        X_p, y_p = [], []
        for i in range(n):
            if not np.isnan(lab[i]):
                X_p.append(emb[i])
                y_p.append(lab[i])

        if len(X_p) < min_residues:
            continue

        X_p = np.stack(X_p).astype(np.float32)
        y_p = np.array(y_p, dtype=np.float64)
        preds = probe.predict(X_p)
        rho, _ = spearmanr(y_p, preds)
        results[pid] = float(rho) if not np.isnan(rho) else 0.0
    return results


def pooled_rho(embs: dict, scores: dict, protein_ids: list,
               probe, max_len: int = 512) -> float:
    """Compute pooled residue-level Spearman rho (headline metric)."""
    all_true, all_pred = [], []
    for pid in protein_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))

        for i in range(n):
            if not np.isnan(lab[i]):
                all_true.append(lab[i])
                all_pred.append(0.0)  # placeholder

        # Predict in batch per protein
        X_p = []
        valid_idx = []
        for i in range(n):
            if not np.isnan(lab[i]):
                X_p.append(emb[i])
                valid_idx.append(len(all_pred) - (n - i))  # will fix below

    # Simpler approach: collect all at once
    all_true_arr, all_pred_arr = [], []
    for pid in protein_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))
        X_p, y_p = [], []
        for i in range(n):
            if not np.isnan(lab[i]):
                X_p.append(emb[i])
                y_p.append(lab[i])
        if len(X_p) == 0:
            continue
        X_p = np.stack(X_p).astype(np.float32)
        preds = probe.predict(X_p)
        all_true_arr.extend(y_p)
        all_pred_arr.extend(preds.tolist())

    rho, _ = spearmanr(all_true_arr, all_pred_arr)
    return float(rho) if not np.isnan(rho) else 0.0


def per_protein_predictions(embs: dict, scores: dict, protein_ids: list,
                            probe, max_len: int = 512,
                            min_residues: int = 5) -> dict:
    """Collect per-protein {y_true, y_pred} arrays from a fitted probe."""
    results = {}
    for pid in protein_ids:
        emb = embs[pid][:max_len]
        lab = np.asarray(scores[pid], dtype=np.float64)[:max_len]
        n = min(len(emb), len(lab))
        X_p, y_p = [], []
        for i in range(n):
            if not np.isnan(lab[i]):
                X_p.append(emb[i])
                y_p.append(lab[i])
        if len(X_p) < min_residues:
            continue
        X_p = np.stack(X_p).astype(np.float32)
        y_p = np.array(y_p, dtype=np.float64)
        preds = probe.predict(X_p)
        results[pid] = {"y_true": y_p, "y_pred": preds}
    return results


# ── Protein characterization ─────────────────────────────────────────

def characterize_protein(scores: dict, pid: str,
                         threshold: float = 8.0) -> dict:
    """Compute properties of a protein's disorder profile."""
    z = np.asarray(scores[pid], dtype=np.float64)
    valid = z[~np.isnan(z)]
    n = len(valid)
    if n == 0:
        return {"pid": pid, "n_residues": 0}

    n_disordered = int(np.sum(valid < threshold))
    return {
        "pid": pid,
        "n_residues": n,
        "frac_disordered": n_disordered / n,
        "frac_ordered": 1 - n_disordered / n,
        "z_mean": float(np.mean(valid)),
        "z_std": float(np.std(valid)),
        "z_min": float(np.min(valid)),
        "z_max": float(np.max(valid)),
        "z_median": float(np.median(valid)),
        "is_bimodal": bool(np.std(valid) > 4.0),  # rough heuristic
    }
```

- [ ] **Step 2: Verify module imports cleanly**

Run:
```bash
uv run python -c "from experiments.disorder_helpers_check import *; print('OK')" 2>&1 || \
uv run python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
import importlib.util
spec = importlib.util.spec_from_file_location('h', 'experiments/45_disorder_helpers.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('Functions:', [x for x in dir(mod) if not x.startswith('_')])
"
```

Expected: List of function names, no import errors.

- [ ] **Step 3: Commit**

```bash
git add experiments/45_disorder_helpers.py
git commit -m "feat(exp45): disorder forensics helper module — data loading, stage transforms, per-protein rho"
```

---

### Task 2: Notebook Layer 1 — Per-Stage Rho Decomposition

**Files:**
- Create: `experiments/45_disorder_forensics.ipynb`

This is the core investigation. For every protein in CheZOD-117 and TriZOD-348 test sets, compute disorder Spearman rho at each pipeline stage: raw → ABTT → RP → PQ.

- [ ] **Step 1: Create notebook with Layer 1 — CheZOD per-stage analysis**

Create `experiments/45_disorder_forensics.ipynb` with these cells:

**Cell 1 — Setup:**
```python
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

from experiments import __path__ as _  # ensure path
sys.path.insert(0, str(Path("experiments").resolve()))

from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("helpers", "experiments/45_disorder_helpers.py")
h = module_from_spec(_spec)
_spec.loader.exec_module(h)

from src.one_embedding.preprocessing import compute_corpus_stats
from src.one_embedding.codec_v2 import OneEmbeddingCodec

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
```

**Cell 2 — Load CheZOD data:**
```python
print("Loading CheZOD data...")
embs, scores, train_ids, test_ids = h.load_chezod_data()
print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")
print(f"  Embedding dim: {next(iter(embs.values())).shape[1]}")
```

**Cell 3 — Compute corpus stats and build pipeline stages:**
```python
# Fit corpus stats on training set
train_embs = {k: embs[k] for k in train_ids}
corpus_stats = compute_corpus_stats(train_embs, n_sample=50_000, n_pcs=5, seed=42)
print(f"Mean vec shape: {corpus_stats['mean_vec'].shape}")
print(f"Top PCs shape: {corpus_stats['top_pcs'].shape}")
print(f"Explained variance (top 5): {corpus_stats['explained_variance']}")

# Build embeddings at each pipeline stage
stages = {}
stages["raw"] = embs
print("Applying ABTT3...")
stages["abtt"] = h.apply_abtt(embs, corpus_stats, k=3)
print("Applying RP 768d...")
stages["rp768"] = h.apply_rp(stages["abtt"], d_out=768, seed=42)

# For PQ: need a fitted codec
print("Fitting PQ codec...")
codec = OneEmbeddingCodec(d_out=768, quantization="pq", pq_m=192)
codec.fit(train_embs)
# Apply RP first (PQ operates on projected space)
stages["pq"] = h.apply_pq(stages["rp768"], codec)

print(f"Stages: {list(stages.keys())}")
for name, st in stages.items():
    sample = next(iter(st.values()))
    print(f"  {name}: dim={sample.shape[1]}")
```

**Cell 4 — Train per-stage probes and compute per-protein rho:**
```python
# Train a Ridge probe at each stage (on training data at that stage)
probes = {}
per_protein_rhos = {}  # {stage: {pid: rho}}
pooled_rhos = {}  # {stage: float}

for stage_name, stage_embs in stages.items():
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"{'='*60}")

    probe = h.train_ridge_probe(stage_embs, scores, train_ids)
    probes[stage_name] = probe
    print(f"  Best alpha: {probe.alpha_}")

    rhos = h.per_protein_rho(stage_embs, scores, test_ids, probe)
    per_protein_rhos[stage_name] = rhos
    print(f"  Per-protein rho: mean={np.mean(list(rhos.values())):.4f}, "
          f"median={np.median(list(rhos.values())):.4f}")

    prho = h.pooled_rho(stage_embs, scores, test_ids, probe)
    pooled_rhos[stage_name] = prho
    print(f"  Pooled rho: {prho:.4f}")

print(f"\n{'='*60}")
print("SUMMARY — Pooled Spearman rho by stage:")
for stage, rho in pooled_rhos.items():
    print(f"  {stage:8s}: {rho:.4f}")
```

**Cell 5 — Compute per-protein deltas:**
```python
# Compute rho drop at each stage for every protein
common_pids = sorted(set.intersection(*[set(r.keys()) for r in per_protein_rhos.values()]))
print(f"Proteins with rho at all stages: {len(common_pids)}")

deltas = {}  # {pid: {delta_abtt, delta_rp, delta_pq, delta_total}}
for pid in common_pids:
    raw_rho = per_protein_rhos["raw"][pid]
    abtt_rho = per_protein_rhos["abtt"][pid]
    rp_rho = per_protein_rhos["rp768"][pid]
    pq_rho = per_protein_rhos["pq"][pid]

    deltas[pid] = {
        "raw_rho": raw_rho,
        "final_rho": pq_rho,
        "delta_abtt": abtt_rho - raw_rho,
        "delta_rp": rp_rho - abtt_rho,
        "delta_pq": pq_rho - rp_rho,
        "delta_total": pq_rho - raw_rho,
    }

# Summary statistics
delta_arr = np.array([d["delta_total"] for d in deltas.values()])
print(f"\nTotal rho change (raw → PQ decoded):")
print(f"  Mean: {np.mean(delta_arr):+.4f}")
print(f"  Median: {np.median(delta_arr):+.4f}")
print(f"  Worst: {np.min(delta_arr):+.4f}")
print(f"  Best: {np.max(delta_arr):+.4f}")

# Which stage causes the most damage?
abtt_drops = np.array([d["delta_abtt"] for d in deltas.values()])
rp_drops = np.array([d["delta_rp"] for d in deltas.values()])
pq_drops = np.array([d["delta_pq"] for d in deltas.values()])
print(f"\nMean rho change per stage:")
print(f"  ABTT:  {np.mean(abtt_drops):+.4f}")
print(f"  RP:    {np.mean(rp_drops):+.4f}")
print(f"  PQ:    {np.mean(pq_drops):+.4f}")
```

**Cell 6 — Visualization: per-stage rho distributions:**
```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
stage_names = ["raw", "abtt", "rp768", "pq"]
stage_labels = ["Raw 1024d", "+ ABTT3", "+ RP 768d", "+ PQ M=192"]

for ax, sname, slabel in zip(axes, stage_names, stage_labels):
    vals = [per_protein_rhos[sname][p] for p in common_pids]
    ax.hist(vals, bins=25, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(vals), color="red", linestyle="--", label=f"mean={np.mean(vals):.3f}")
    ax.axvline(np.median(vals), color="blue", linestyle=":", label=f"med={np.median(vals):.3f}")
    ax.set_title(slabel)
    ax.set_xlabel("Per-protein Spearman ρ")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Count")
plt.suptitle("CheZOD-117: Per-Protein Disorder ρ at Each Pipeline Stage", fontsize=13)
plt.tight_layout()
plt.savefig("../results/disorder_forensics/chezod_per_stage_rho.png", dpi=150)
plt.show()
```

**Cell 7 — Visualization: stage-wise rho drops:**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
drop_data = [
    (abtt_drops, "ABTT3", "tab:blue"),
    (rp_drops, "RP 768d", "tab:orange"),
    (pq_drops, "PQ M=192", "tab:green"),
]
for ax, (drops, label, color) in zip(axes, drop_data):
    ax.hist(drops, bins=25, alpha=0.7, color=color, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(np.mean(drops), color="red", linestyle="--", label=f"mean={np.mean(drops):+.3f}")
    ax.set_title(f"Δρ from {label}")
    ax.set_xlabel("Per-protein ρ change")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Count")
plt.suptitle("CheZOD-117: Per-Protein ρ Change at Each Stage", fontsize=13)
plt.tight_layout()
plt.savefig("../results/disorder_forensics/chezod_stage_drops.png", dpi=150)
plt.show()
```

- [ ] **Step 2: Create output directory**

```bash
mkdir -p results/disorder_forensics
```

- [ ] **Step 3: Commit**

```bash
git add experiments/45_disorder_forensics.ipynb results/disorder_forensics/.gitkeep
git commit -m "feat(exp45): disorder forensics notebook — Layer 1 per-stage rho decomposition"
```

---

### Task 3: Notebook Layer 1b — TriZOD + Combined Analysis

**Files:**
- Modify: `experiments/45_disorder_forensics.ipynb`

Add TriZOD analysis and the combined ranking.

- [ ] **Step 1: Add TriZOD cells to notebook**

**Cell 8 — Load TriZOD and repeat per-stage analysis:**
```python
print("Loading TriZOD data...")
tz_embs, tz_scores, tz_train_ids, tz_test_ids = h.load_trizod_data()
print(f"  Train: {len(tz_train_ids)}, Test: {len(tz_test_ids)}")

# Fit corpus stats on TriZOD training set
tz_train_embs = {k: tz_embs[k] for k in tz_train_ids if k in tz_embs}
tz_corpus_stats = compute_corpus_stats(tz_train_embs, n_sample=50_000, n_pcs=5, seed=42)

# Build stages
tz_stages = {}
tz_stages["raw"] = tz_embs
tz_stages["abtt"] = h.apply_abtt(tz_embs, tz_corpus_stats, k=3)
tz_stages["rp768"] = h.apply_rp(tz_stages["abtt"], d_out=768, seed=42)

tz_codec = OneEmbeddingCodec(d_out=768, quantization="pq", pq_m=192)
tz_codec.fit(tz_train_embs)
tz_stages["pq"] = h.apply_pq(tz_stages["rp768"], tz_codec)

# Train probes and compute rhos
tz_per_protein_rhos = {}
tz_pooled_rhos = {}
for stage_name, stage_embs in tz_stages.items():
    probe = h.train_ridge_probe(stage_embs, tz_scores, tz_train_ids)
    tz_per_protein_rhos[stage_name] = h.per_protein_rho(
        stage_embs, tz_scores, tz_test_ids, probe
    )
    tz_pooled_rhos[stage_name] = h.pooled_rho(
        stage_embs, tz_scores, tz_test_ids, probe
    )

print("\nTriZOD — Pooled Spearman rho by stage:")
for stage, rho in tz_pooled_rhos.items():
    print(f"  {stage:8s}: {rho:.4f}")
```

**Cell 9 — Combined CheZOD + TriZOD summary table:**
```python
print(f"{'Stage':<10} {'CheZOD pooled ρ':>16} {'TriZOD pooled ρ':>16}")
print("-" * 44)
for stage in ["raw", "abtt", "rp768", "pq"]:
    print(f"{stage:<10} {pooled_rhos[stage]:>16.4f} {tz_pooled_rhos[stage]:>16.4f}")

print(f"\n{'Stage':<10} {'CheZOD Δρ':>16} {'TriZOD Δρ':>16}")
print("-" * 44)
prev_cz, prev_tz = pooled_rhos["raw"], tz_pooled_rhos["raw"]
for stage in ["abtt", "rp768", "pq"]:
    cz_delta = pooled_rhos[stage] - prev_cz
    tz_delta = tz_pooled_rhos[stage] - prev_tz
    print(f"{stage:<10} {cz_delta:>+16.4f} {tz_delta:>+16.4f}")
    prev_cz, prev_tz = pooled_rhos[stage], tz_pooled_rhos[stage]
```

- [ ] **Step 2: Commit**

```bash
git add experiments/45_disorder_forensics.ipynb
git commit -m "feat(exp45): Layer 1b — TriZOD per-stage analysis + combined summary"
```

---

### Task 4: Notebook Layer 2 — Stratified Protein Selection

**Files:**
- Modify: `experiments/45_disorder_forensics.ipynb`

Select proteins for deep analysis: average, worst losers, surprising winners.

- [ ] **Step 1: Add protein selection and characterization cells**

**Cell 10 — Rank proteins and select strata:**
```python
# CheZOD: rank by total rho drop
sorted_by_drop = sorted(deltas.items(), key=lambda x: x[1]["delta_total"])
worst_3 = [pid for pid, _ in sorted_by_drop[:3]]
best_3 = [pid for pid, _ in sorted_by_drop[-3:]]

# Median proteins
drops = [d["delta_total"] for d in deltas.values()]
median_drop = np.median(drops)
by_distance_to_median = sorted(
    deltas.items(), key=lambda x: abs(x[1]["delta_total"] - median_drop)
)
average_3 = [pid for pid, _ in by_distance_to_median[:3]]

selected_cz = {
    "worst": worst_3,
    "average": average_3,
    "winners": best_3,
}

print("CheZOD selected proteins:")
for category, pids in selected_cz.items():
    print(f"\n  {category.upper()}:")
    for pid in pids:
        d = deltas[pid]
        print(f"    {pid}: raw_ρ={d['raw_rho']:.3f} → final_ρ={d['final_rho']:.3f} "
              f"(Δ={d['delta_total']:+.3f})  "
              f"[ABTT:{d['delta_abtt']:+.3f} RP:{d['delta_rp']:+.3f} PQ:{d['delta_pq']:+.3f}]")
```

**Cell 11 — Characterize selected proteins:**
```python
print(f"\n{'PID':<12} {'Cat':<8} {'Len':>4} {'%Dis':>5} {'Z_mean':>7} {'Z_std':>6} "
      f"{'Δ_total':>8} {'Δ_ABTT':>7} {'Δ_RP':>7} {'Δ_PQ':>7}")
print("-" * 90)

for category, pids in selected_cz.items():
    for pid in pids:
        d = deltas[pid]
        props = h.characterize_protein(scores, pid)
        print(f"{pid:<12} {category:<8} {props['n_residues']:>4} "
              f"{props['frac_disordered']:>5.1%} {props['z_mean']:>7.1f} {props['z_std']:>6.1f} "
              f"{d['delta_total']:>+8.3f} {d['delta_abtt']:>+7.3f} "
              f"{d['delta_rp']:>+7.3f} {d['delta_pq']:>+7.3f}")
```

**Cell 12 — Do the same for TriZOD:**
```python
# TriZOD protein ranking
tz_common = sorted(set.intersection(*[set(r.keys()) for r in tz_per_protein_rhos.values()]))
tz_deltas = {}
for pid in tz_common:
    tz_deltas[pid] = {
        "raw_rho": tz_per_protein_rhos["raw"][pid],
        "final_rho": tz_per_protein_rhos["pq"][pid],
        "delta_abtt": tz_per_protein_rhos["abtt"][pid] - tz_per_protein_rhos["raw"][pid],
        "delta_rp": tz_per_protein_rhos["rp768"][pid] - tz_per_protein_rhos["abtt"][pid],
        "delta_pq": tz_per_protein_rhos["pq"][pid] - tz_per_protein_rhos["rp768"][pid],
        "delta_total": tz_per_protein_rhos["pq"][pid] - tz_per_protein_rhos["raw"][pid],
    }

tz_sorted = sorted(tz_deltas.items(), key=lambda x: x[1]["delta_total"])
tz_worst_3 = [pid for pid, _ in tz_sorted[:3]]
tz_best_3 = [pid for pid, _ in tz_sorted[-3:]]
tz_drops = [d["delta_total"] for d in tz_deltas.values()]
tz_median_drop = np.median(tz_drops)
tz_by_median = sorted(tz_deltas.items(), key=lambda x: abs(x[1]["delta_total"] - tz_median_drop))
tz_average_3 = [pid for pid, _ in tz_by_median[:3]]

selected_tz = {"worst": tz_worst_3, "average": tz_average_3, "winners": tz_best_3}

print("TriZOD selected proteins:")
for category, pids in selected_tz.items():
    print(f"\n  {category.upper()}:")
    for pid in pids:
        d = tz_deltas[pid]
        print(f"    {pid}: raw_ρ={d['raw_rho']:.3f} → final_ρ={d['final_rho']:.3f} "
              f"(Δ={d['delta_total']:+.3f})")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/45_disorder_forensics.ipynb
git commit -m "feat(exp45): Layer 2 — stratified protein selection (worst/average/winners)"
```

---

### Task 5: Notebook Layer 3 — Subspace Forensics

**Files:**
- Modify: `experiments/45_disorder_forensics.ipynb`

Identify which directions in embedding space carry disorder signal and whether ABTT/RP destroy them.

- [ ] **Step 1: Add subspace analysis cells**

**Cell 13 — Find disorder-informative dimensions:**
```python
# Stack all training residues with valid Z-scores
X_all, y_all = [], []
for pid in train_ids:
    emb = embs[pid][:512]
    lab = np.asarray(scores[pid], dtype=np.float64)[:512]
    n = min(len(emb), len(lab))
    for i in range(n):
        if not np.isnan(lab[i]):
            X_all.append(emb[i])
            y_all.append(lab[i])

X_all = np.stack(X_all).astype(np.float32)
y_all = np.array(y_all, dtype=np.float64)
print(f"Training residues: {len(y_all)}, dimensions: {X_all.shape[1]}")

# Per-dimension correlation with Z-scores
dim_correlations = np.zeros(X_all.shape[1])
for d in range(X_all.shape[1]):
    rho, _ = spearmanr(X_all[:, d], y_all)
    dim_correlations[d] = rho if not np.isnan(rho) else 0.0

# Top disorder-informative dimensions
top_k = 50
top_dims = np.argsort(np.abs(dim_correlations))[::-1][:top_k]
print(f"\nTop {top_k} disorder-correlated dimensions:")
print(f"  Max |ρ|: {np.abs(dim_correlations[top_dims[0]]):.4f} (dim {top_dims[0]})")
print(f"  Min |ρ| in top-{top_k}: {np.abs(dim_correlations[top_dims[-1]]):.4f}")
print(f"  Mean |ρ| across all dims: {np.mean(np.abs(dim_correlations)):.4f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(1024), np.abs(dim_correlations), width=1, alpha=0.5)
ax.bar(top_dims, np.abs(dim_correlations[top_dims]), width=1, color="red", alpha=0.8)
ax.set_xlabel("Dimension index")
ax.set_ylabel("|Spearman ρ| with Z-score")
ax.set_title("Per-Dimension Disorder Correlation (red = top 50)")
plt.tight_layout()
plt.savefig("../results/disorder_forensics/dimension_disorder_correlation.png", dpi=150)
plt.show()
```

**Cell 14 — ABTT overlap: do removed PCs carry disorder signal?**
```python
# The top-3 PCs removed by ABTT
top3_pcs = corpus_stats["top_pcs"][:3]  # (3, 1024)

# Project disorder-correlated direction onto PC space
# Disorder direction = correlation vector normalized
disorder_direction = dim_correlations / (np.linalg.norm(dim_correlations) + 1e-10)

# Cosine similarity of each PC with the disorder direction
for i in range(3):
    pc = top3_pcs[i]
    cos_sim = np.dot(pc, disorder_direction) / (np.linalg.norm(pc) + 1e-10)
    # Also: how much variance in the disorder-informative dims do these PCs explain?
    proj_on_top_dims = np.sum(pc[top_dims] ** 2) / np.sum(pc ** 2)
    print(f"PC{i+1}: cos_sim with disorder direction = {cos_sim:+.4f}, "
          f"weight on top-50 disorder dims = {proj_on_top_dims:.1%}")

# Compare with PCs 4-5 (not removed)
for i in range(3, 5):
    pc = corpus_stats["top_pcs"][i]
    cos_sim = np.dot(pc, disorder_direction) / (np.linalg.norm(pc) + 1e-10)
    print(f"PC{i+1} (kept): cos_sim = {cos_sim:+.4f}")

# Variance of disorder-informative dims before/after ABTT
raw_sample = np.stack([embs[pid][:512] for pid in test_ids[:20]])
raw_sample = raw_sample.reshape(-1, 1024)
abtt_sample = np.stack([stages["abtt"][pid][:512] for pid in test_ids[:20]])
abtt_sample = abtt_sample.reshape(-1, raw_sample.shape[1])  # might be diff shape

var_raw = np.var(raw_sample[:, top_dims], axis=0).sum()
var_abtt = np.var(abtt_sample[:, top_dims], axis=0).sum()
print(f"\nVariance in top-50 disorder dims: raw={var_raw:.2f}, ABTT={var_abtt:.2f}, "
      f"retained={var_abtt/var_raw:.1%}")
```

**Cell 15 — RP preservation: how much disorder subspace survives projection?**
```python
# RP projects 1024d → 768d. The disorder-informative subspace is defined by
# the top-50 dimensions. How much of their variance is preserved?

# Build the projection matrix (same as codec)
rng = np.random.RandomState(42)
R = rng.randn(1024, 768).astype(np.float32)
Q, _ = np.linalg.qr(R, mode="reduced")
proj_matrix = Q * np.sqrt(1024 / 768)

# For each of the top-50 disorder dims, compute how much of that
# dimension's unit vector is preserved after projection
preservation = np.zeros(top_k)
for idx, d in enumerate(top_dims):
    e_d = np.zeros(1024, dtype=np.float32)
    e_d[d] = 1.0
    projected = e_d @ proj_matrix  # (768,)
    preservation[idx] = np.linalg.norm(projected) ** 2  # fraction of variance

print(f"RP preservation of disorder-informative dims:")
print(f"  Mean preservation: {np.mean(preservation):.4f} (expected ~{768/1024:.4f} = d_out/d_in)")
print(f"  Min: {np.min(preservation):.4f}, Max: {np.max(preservation):.4f}")
print(f"  RP preserves {768/1024:.1%} of variance on average (Johnson-Lindenstrauss)")

# Key question: is disorder preservation worse than average?
all_preservation = np.array([
    np.linalg.norm(np.eye(1024, dtype=np.float32)[d] @ proj_matrix) ** 2
    for d in range(1024)
])
print(f"\n  All-dimension mean preservation: {np.mean(all_preservation):.4f}")
print(f"  Disorder-dim mean preservation: {np.mean(preservation):.4f}")
print(f"  Difference: {np.mean(preservation) - np.mean(all_preservation):+.4f}")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/45_disorder_forensics.ipynb
git commit -m "feat(exp45): Layer 3 — subspace forensics (ABTT overlap, RP preservation)"
```

---

### Task 6: Notebook Layer 4 — Residue-Level Error Maps

**Files:**
- Modify: `experiments/45_disorder_forensics.ipynb`

For each selected protein, plot residue-level true vs predicted Z-scores and embedding distortion.

- [ ] **Step 1: Add residue-level diagnostic plots**

**Cell 16 — Collect per-protein predictions at raw and final stages:**
```python
# Get predictions for selected CheZOD proteins at raw and PQ stages
raw_preds = h.per_protein_predictions(stages["raw"], scores,
    [p for cat in selected_cz.values() for p in cat],
    probes["raw"])
pq_preds = h.per_protein_predictions(stages["pq"], scores,
    [p for cat in selected_cz.values() for p in cat],
    probes["pq"])
```

**Cell 17 — Residue trace plots for selected proteins:**
```python
all_selected = [(cat, pid) for cat, pids in selected_cz.items() for pid in pids]

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, (category, pid) in enumerate(all_selected):
    ax = axes[idx]
    if pid not in raw_preds or pid not in pq_preds:
        ax.set_title(f"{pid} — no data")
        continue

    y_true = raw_preds[pid]["y_true"]
    y_raw = raw_preds[pid]["y_pred"]
    y_pq = pq_preds[pid]["y_pred"]
    x = np.arange(len(y_true))

    ax.plot(x, y_true, "k-", alpha=0.7, linewidth=0.8, label="True Z")
    ax.plot(x, y_raw, "b-", alpha=0.6, linewidth=0.8, label="Raw probe")
    ax.plot(x, y_pq, "r-", alpha=0.6, linewidth=0.8, label="PQ probe")
    ax.axhline(8.0, color="gray", linestyle=":", linewidth=0.5, label="Z=8 threshold")

    # Highlight residues where raw is better by >2 Z-score units
    error_raw = np.abs(y_true - y_raw)
    error_pq = np.abs(y_true - y_pq)
    worse = (error_pq - error_raw) > 2.0
    if worse.any():
        ax.scatter(x[worse], y_true[worse], color="red", s=10, zorder=5, label="Codec loss")

    d = deltas[pid]
    props = h.characterize_protein(scores, pid)
    ax.set_title(f"{pid} [{category}] Δρ={d['delta_total']:+.3f} "
                 f"({props['n_residues']}res, {props['frac_disordered']:.0%}dis)", fontsize=9)
    if idx == 0:
        ax.legend(fontsize=7)
    ax.set_xlabel("Residue" if idx >= 6 else "")
    ax.set_ylabel("Z-score" if idx % 3 == 0 else "")

plt.suptitle("CheZOD: True Z vs Predicted (Raw=blue, Compressed=red)", fontsize=13)
plt.tight_layout()
plt.savefig("../results/disorder_forensics/chezod_residue_traces.png", dpi=150)
plt.show()
```

**Cell 18 — Embedding distortion vs prediction error:**
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
categories_to_plot = ["worst", "average", "winners"]

for ax, category in zip(axes, categories_to_plot):
    for pid in selected_cz[category]:
        # L2 distance between raw and compressed embeddings per residue
        raw_emb = stages["raw"][pid][:512]
        pq_emb = stages["pq"][pid][:512]
        # Need same dimensionality — compare at RP stage vs PQ stage
        rp_emb = stages["rp768"][pid][:512]
        pq_emb_aligned = stages["pq"][pid][:512]

        n = min(len(rp_emb), len(pq_emb_aligned))
        distortion = np.linalg.norm(rp_emb[:n] - pq_emb_aligned[:n], axis=1)

        # Prediction error increase
        if pid in raw_preds and pid in pq_preds:
            y_true = raw_preds[pid]["y_true"]
            err_raw = np.abs(raw_preds[pid]["y_pred"] - y_true)
            err_pq = np.abs(pq_preds[pid]["y_pred"] - y_true)
            delta_err = err_pq - err_raw

            n_plot = min(len(distortion), len(delta_err))
            ax.scatter(distortion[:n_plot], delta_err[:n_plot],
                      alpha=0.3, s=5, label=pid)

    ax.set_xlabel("Embedding L2 distortion (RP→PQ)")
    ax.set_ylabel("Prediction error increase")
    ax.set_title(f"{category.upper()} proteins")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=7)

plt.suptitle("Embedding Distortion vs Prediction Error", fontsize=13)
plt.tight_layout()
plt.savefig("../results/disorder_forensics/distortion_vs_error.png", dpi=150)
plt.show()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/45_disorder_forensics.ipynb
git commit -m "feat(exp45): Layer 4 — residue-level error maps and distortion analysis"
```

---

### Task 7: Notebook Layer 5 — Targeted Recovery Experiments

**Files:**
- Modify: `experiments/45_disorder_forensics.ipynb`

Test specific hypotheses based on Layer 1-4 findings.

- [ ] **Step 1: Add recovery experiment cells**

**Cell 19 — ABTT k sweep:**
```python
# Does removing fewer PCs help disorder?
print("ABTT k sweep: pooled Spearman rho on CheZOD")
print(f"{'k':>3} {'Pooled ρ':>10} {'Δ vs raw':>10}")
print("-" * 25)

raw_rho = pooled_rhos["raw"]
print(f"{'raw':>3} {raw_rho:>10.4f} {'baseline':>10}")

for k in [0, 1, 2, 3, 4, 5]:
    if k == 0:
        # Just centering, no PC removal
        centered = {}
        for pid, emb in embs.items():
            centered[pid] = center_embeddings(emb, corpus_stats["mean_vec"])
        stage_embs = h.apply_rp(centered, d_out=768, seed=42)
    else:
        abtted = h.apply_abtt(embs, corpus_stats, k=k)
        stage_embs = h.apply_rp(abtted, d_out=768, seed=42)

    probe = h.train_ridge_probe(stage_embs, scores, train_ids)
    prho = h.pooled_rho(stage_embs, scores, test_ids, probe)
    print(f"{k:>3} {prho:>10.4f} {prho - raw_rho:>+10.4f}")
```

**Cell 20 — d_out sweep:**
```python
# Does keeping more dimensions help disorder?
print("\nd_out sweep: pooled Spearman rho on CheZOD (ABTT k=3)")
print(f"{'d_out':>6} {'Pooled ρ':>10} {'Δ vs raw':>10} {'Dims lost':>10}")
print("-" * 40)

for d_out in [1024, 896, 832, 768, 640, 512]:
    if d_out >= 1024:
        # No RP
        stage_embs = stages["abtt"]
    else:
        stage_embs = h.apply_rp(stages["abtt"], d_out=d_out, seed=42)

    probe = h.train_ridge_probe(stage_embs, scores, train_ids)
    prho = h.pooled_rho(stage_embs, scores, test_ids, probe)
    print(f"{d_out:>6} {prho:>10.4f} {prho - raw_rho:>+10.4f} {1024 - d_out:>10}")
```

**Cell 21 — fp16 at 768d (no PQ) to isolate PQ contribution:**
```python
# Is the gap from RP or PQ at the final stage?
rp_fp16 = h.apply_fp16(stages["rp768"])
probe_fp16 = h.train_ridge_probe(rp_fp16, scores, train_ids)
rho_fp16 = h.pooled_rho(rp_fp16, scores, test_ids, probe_fp16)

print(f"\nIsolating PQ contribution at 768d:")
print(f"  RP 768d (float32): {pooled_rhos['rp768']:.4f}")
print(f"  RP 768d (fp16):    {rho_fp16:.4f}")
print(f"  RP 768d + PQ:      {pooled_rhos['pq']:.4f}")
print(f"  PQ-only drop:      {pooled_rhos['pq'] - pooled_rhos['rp768']:+.4f}")
print(f"  fp16-only drop:    {rho_fp16 - pooled_rhos['rp768']:+.4f}")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/45_disorder_forensics.ipynb
git commit -m "feat(exp45): Layer 5 — recovery experiments (ABTT k sweep, d_out sweep, fp16 isolation)"
```

---

### Task 8: Summary Cell and Final Cleanup

**Files:**
- Modify: `experiments/45_disorder_forensics.ipynb`

Add a summary cell that collects all findings.

- [ ] **Step 1: Add summary cell**

**Cell 22 — Executive summary:**
```python
print("=" * 70)
print("DISORDER INFORMATION LOSS — INVESTIGATION SUMMARY")
print("=" * 70)

print(f"\n1. PER-STAGE DECOMPOSITION (CheZOD pooled ρ):")
for stage, rho in pooled_rhos.items():
    print(f"   {stage:8s}: {rho:.4f}")

print(f"\n2. MEAN PER-PROTEIN ρ CHANGE BY STAGE:")
print(f"   ABTT:  {np.mean(abtt_drops):+.4f}")
print(f"   RP:    {np.mean(rp_drops):+.4f}")
print(f"   PQ:    {np.mean(pq_drops):+.4f}")

print(f"\n3. PROTEIN SELECTION:")
print(f"   Worst losers:  {selected_cz['worst']}")
print(f"   Average:        {selected_cz['average']}")
print(f"   Winners:        {selected_cz['winners']}")

print(f"\n4. SUBSPACE ANALYSIS:")
print(f"   (Review Cell 14-15 outputs)")

print(f"\n5. RECOVERY EXPERIMENTS:")
print(f"   (Review Cell 19-21 outputs)")

print(f"\n{'='*70}")
print("NEXT STEPS:")
print("  - Extract ESM2-650M + ESM-C 650M embeddings for CheZOD + TriZOD")
print("  - Convert findings into experiments/45_disorder_forensics.py")
print("  - Run validation script on all 3 PLMs")
print(f"{'='*70}")
```

- [ ] **Step 2: Final commit**

```bash
git add experiments/45_disorder_forensics.ipynb
git commit -m "feat(exp45): disorder forensics complete — 5-layer investigation notebook"
```

---

Plan complete and saved to `docs/superpowers/plans/2026-03-30-disorder-information-loss.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?