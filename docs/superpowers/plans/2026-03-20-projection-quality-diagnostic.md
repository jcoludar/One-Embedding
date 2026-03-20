# Projection Quality Diagnostic — Experiment 38

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the ~10-15% per-residue quality loss in V2 codec can be recovered by using PCA instead of RP, or by projecting to higher dimensions.

**Architecture:** Single experiment script `experiments/38_projection_quality_diagnostic.py` with 3 steps:
- Step A: RP dimension sweep (256, 384, 512, 640, 768, 896, 1024) — all 4 per-residue tasks + retrieval. Find inflection point where TM/disorder stop degrading.
- Step B: PCA dimension sweep (same dimensions) — same metrics, compare to RP at every point
- Step C: Channel importance analysis — which raw 1024d channels correlate with TM/disorder labels, are they concentrated in specific PCs (informing hybrid projection design)

**Tech Stack:** numpy, sklearn (PCA, LogisticRegression, Ridge), scipy (spearmanr), existing evaluation infrastructure

---

### Task 1: Create experiment script scaffold with data loading

**Files:**
- Create: `experiments/38_projection_quality_diagnostic.py`
- Read: `data/residue_embeddings/prot_t5_xl_medium5k.h5`, `data/residue_embeddings/prot_t5_xl_cb513.h5`

- [ ] **Step 1: Create scaffold with imports, paths, and result checkpointing**

```python
#!/usr/bin/env python3
"""Experiment 38: Projection quality diagnostic.

Traces per-residue quality loss through RP vs PCA at multiple dimensions.
Identifies which raw embedding channels carry TM/disorder signal.

Steps:
    A: RP dimension sweep (512, 640, 768) — SS3, SS8, disorder, TM, retrieval
    B: PCA dimension sweep (512, 640, 768) — same metrics
    C: Channel importance for TM and disorder
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.preprocessing import compute_corpus_stats, all_but_the_top
from src.one_embedding.universal_transforms import random_orthogonal_project
from src.one_embedding.transforms import dct_summary
from src.evaluation.per_residue_tasks import (
    evaluate_ss3_probe, evaluate_ss8_probe,
    evaluate_disorder_probe, evaluate_tm_probe,
    load_cb513_csv, load_chezod_seth, load_tmbed_annotated,
)
from src.evaluation.retrieval import evaluate_retrieval_from_vectors
from src.extraction.data_loader import load_metadata_csv, filter_by_family_size
from src.utils.h5_store import load_residue_embeddings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_PATH = DATA_DIR / "benchmarks" / "projection_diagnostic_results.json"

EMB_5K = DATA_DIR / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
EMB_CB513 = DATA_DIR / "residue_embeddings" / "prot_t5_xl_cb513.h5"
EMB_CHEZOD = DATA_DIR / "residue_embeddings" / "prot_t5_xl_chezod.h5"
EMB_VALID = DATA_DIR / "residue_embeddings" / "prot_t5_xl_validation.h5"
META_5K = DATA_DIR / "proteins" / "metadata_5k.csv"
SPLIT_5K = DATA_DIR / "splits" / "esm2_650m_5k_split.json"
CB513_CSV = DATA_DIR / "per_residue_benchmarks" / "CB513.csv"
TMBED_FASTA = DATA_DIR / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"

DIMS = [256, 384, 512, 640, 768, 896, 1024]


def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"steps_done": []}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
```

- [ ] **Step 2: Add helper functions for projection and evaluation**

```python
def project_rp(embeddings: dict, d_out: int, seed: int = 42) -> dict:
    """Apply random orthogonal projection to all proteins."""
    result = {}
    proj_mat = None
    for pid, emb in embeddings.items():
        if proj_mat is None:
            # Build projection matrix once
            rng = np.random.RandomState(seed)
            D = emb.shape[1]
            R = rng.randn(D, d_out).astype(np.float32)
            Q, _ = np.linalg.qr(R, mode="reduced")
            proj_mat = Q * np.sqrt(D / d_out)
        result[pid] = (emb.astype(np.float32) @ proj_mat).astype(np.float32)
    return result


def project_pca(embeddings: dict, rotation_matrix: np.ndarray,
                mean_vec: np.ndarray, d_out: int) -> dict:
    """Apply PCA projection: center, rotate, truncate to top d_out components."""
    D = rotation_matrix.shape[1]
    if d_out >= D:
        # Full PCA rotation, no truncation
        R = rotation_matrix.T
    else:
        R = rotation_matrix[:d_out].T  # (D, d_out) — top d_out PCs as columns
    result = {}
    for pid, emb in embeddings.items():
        centered = emb.astype(np.float32) - mean_vec
        result[pid] = (centered @ R).astype(np.float32)
    return result


def evaluate_all_per_residue(embeddings_cb513, embeddings_chezod, embeddings_tm,
                              ss3_labels, ss8_labels,
                              disorder_scores, disorder_train, disorder_test,
                              tm_labels, tm_train, tm_test,
                              cb_train, cb_test):
    """Run all 4 per-residue probes. Returns dict of metrics."""
    results = {}

    ss3 = evaluate_ss3_probe(embeddings_cb513, ss3_labels, cb_train, cb_test)
    results["ss3_q3"] = ss3["q3"]

    ss8 = evaluate_ss8_probe(embeddings_cb513, ss8_labels, cb_train, cb_test)
    results["ss8_q8"] = ss8["q8"]

    disorder = evaluate_disorder_probe(
        embeddings_chezod, disorder_scores, disorder_train, disorder_test
    )
    results["disorder_rho"] = disorder["spearman_rho"]

    tm = evaluate_tm_probe(embeddings_tm, tm_labels, tm_train, tm_test)
    results["tm_f1"] = tm["macro_f1"]

    return results


def evaluate_retrieval(embeddings_5k, metadata, test_ids, d_out):
    """Compute protein vectors via DCT K=4 and evaluate retrieval."""
    vecs = {}
    for pid, emb in embeddings_5k.items():
        vecs[pid] = dct_summary(emb, K=4)
    ret = evaluate_retrieval_from_vectors(
        vecs, metadata, label_key="family",
        query_ids=test_ids, database_ids=test_ids,
    )
    return {"family_ret1": ret["precision@1"], "mrr": ret["mrr"]}
```

- [ ] **Step 3: Verify script loads and imports work**

Run: `uv run python experiments/38_projection_quality_diagnostic.py --help 2>&1 || uv run python -c "exec(open('experiments/38_projection_quality_diagnostic.py').read().split('if __name__')[0])"`
Expected: No import errors

- [ ] **Step 4: Commit scaffold**

```bash
git add experiments/38_projection_quality_diagnostic.py
git commit -m "feat(exp38): scaffold for projection quality diagnostic"
```

---

### Task 2: Step A — RP dimension sweep

**Files:**
- Modify: `experiments/38_projection_quality_diagnostic.py`

- [ ] **Step 1: Implement step_A function**

```python
def step_A(results):
    """RP dimension sweep: 512, 640, 768 — all per-residue + retrieval."""
    if "A" in results.get("steps_done", []):
        print("Step A already done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step A: RP Dimension Sweep")
    print("=" * 60)

    # Load data
    print("  Loading embeddings...")
    emb_5k = load_residue_embeddings(EMB_5K)
    emb_cb513 = load_residue_embeddings(EMB_CB513)

    # Try loading chezod and validation embeddings
    emb_chezod = {}
    if EMB_CHEZOD.exists():
        emb_chezod = load_residue_embeddings(EMB_CHEZOD)
    if EMB_VALID.exists():
        emb_valid = load_residue_embeddings(EMB_VALID)
        # Merge chezod-prefixed keys from validation
        for k, v in emb_valid.items():
            if k.startswith("chezod_") or k.startswith("tmbed_"):
                emb_chezod[k] = v

    # Load labels
    _, ss3_labels, ss8_labels, _ = load_cb513_csv(CB513_CSV)
    _, disorder_scores, disorder_train, disorder_test = load_chezod_seth(
        DATA_DIR / "per_residue_benchmarks"
    )
    _, tm_labels = load_tmbed_annotated(TMBED_FASTA)

    # CB513 split
    cb513_avail = sorted(set(emb_cb513.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]

    # TM split — use validation set proteins
    tm_avail = sorted(set(emb_chezod.keys()) & set(tm_labels.keys()))
    rng2 = random.Random(42)
    rng2.shuffle(tm_avail)
    n_tr_tm = int(len(tm_avail) * 0.8)
    tm_train, tm_test = tm_avail[:n_tr_tm], tm_avail[n_tr_tm:]

    # Retrieval metadata + split
    metadata = load_metadata_csv(META_5K)
    metadata, kept_ids = filter_by_family_size(metadata, min_members=3)
    with open(SPLIT_5K) as f:
        split = json.load(f)
    ret_test_ids = [i for i in split["test_ids"] if i in kept_ids]

    # Compute corpus stats (for ABTT3)
    print("  Computing corpus stats...")
    stats = compute_corpus_stats(emb_5k, n_sample=50_000, n_pcs=5, seed=42)
    top3 = stats["top_pcs"][:3]

    # --- Raw baseline ---
    print("\n  Evaluating raw 1024d baseline...")
    raw_pr = evaluate_all_per_residue(
        emb_cb513, emb_chezod, emb_chezod,
        ss3_labels, ss8_labels,
        disorder_scores, disorder_train, disorder_test,
        tm_labels, tm_train, tm_test,
        cb_train, cb_test,
    )
    print(f"    Raw: SS3={raw_pr['ss3_q3']:.3f} SS8={raw_pr['ss8_q8']:.3f} "
          f"Dis={raw_pr['disorder_rho']:.3f} TM={raw_pr['tm_f1']:.3f}")

    step_results = {"raw_1024d": raw_pr, "rp": {}}

    # --- RP at each dimension ---
    for d_out in DIMS:
        print(f"\n  RP d={d_out}...")
        t0 = time.time()

        # Apply ABTT3 + RP to each dataset (d=1024 means ABTT3 only, no projection)
        def apply_abtt_rp(embs, d):
            projected = {}
            for pid, emb in embs.items():
                e = all_but_the_top(emb.astype(np.float32), top3)
                if d < e.shape[1]:
                    projected[pid] = random_orthogonal_project(e, d_out=d, seed=42)
                else:
                    projected[pid] = e  # No projection needed at full dim
            return projected

        cb_proj = apply_abtt_rp(emb_cb513, d_out)
        cz_proj = apply_abtt_rp(emb_chezod, d_out)
        fk_proj = apply_abtt_rp(emb_5k, d_out)

        # Per-residue
        pr = evaluate_all_per_residue(
            cb_proj, cz_proj, cz_proj,
            ss3_labels, ss8_labels,
            disorder_scores, disorder_train, disorder_test,
            tm_labels, tm_train, tm_test,
            cb_train, cb_test,
        )

        # Retrieval
        ret = evaluate_retrieval(fk_proj, metadata, ret_test_ids, d_out)

        elapsed = time.time() - t0
        step_results["rp"][str(d_out)] = {**pr, **ret, "elapsed_s": round(elapsed, 1)}
        print(f"    RP-{d_out}: SS3={pr['ss3_q3']:.3f} SS8={pr['ss8_q8']:.3f} "
              f"Dis={pr['disorder_rho']:.3f} TM={pr['tm_f1']:.3f} "
              f"Ret@1={ret['family_ret1']:.3f} ({elapsed:.1f}s)")

    results["A"] = step_results
    results["steps_done"].append("A")
    save_results(results)
    print("\n  Step A complete!")
    return results
```

- [ ] **Step 2: Run Step A**

Run: `uv run python experiments/38_projection_quality_diagnostic.py` (after adding `__main__` block in Task 4)
Expected: Results for raw + RP at 512/640/768 with all metrics

- [ ] **Step 3: Commit**

```bash
git add experiments/38_projection_quality_diagnostic.py
git commit -m "feat(exp38): step A — RP dimension sweep with all per-residue tasks"
```

---

### Task 3: Step B — PCA dimension sweep

**Files:**
- Modify: `experiments/38_projection_quality_diagnostic.py`

- [ ] **Step 1: Implement step_B function**

```python
def step_B(results):
    """PCA dimension sweep: 512, 640, 768 — all per-residue + retrieval."""
    if "B" in results.get("steps_done", []):
        print("Step B already done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step B: PCA Dimension Sweep")
    print("=" * 60)

    # Load same data as Step A
    print("  Loading embeddings...")
    emb_5k = load_residue_embeddings(EMB_5K)
    emb_cb513 = load_residue_embeddings(EMB_CB513)

    emb_chezod = {}
    if EMB_CHEZOD.exists():
        emb_chezod = load_residue_embeddings(EMB_CHEZOD)
    if EMB_VALID.exists():
        emb_valid = load_residue_embeddings(EMB_VALID)
        for k, v in emb_valid.items():
            if k.startswith("chezod_") or k.startswith("tmbed_"):
                emb_chezod[k] = v

    _, ss3_labels, ss8_labels, _ = load_cb513_csv(CB513_CSV)
    _, disorder_scores, disorder_train, disorder_test = load_chezod_seth(
        DATA_DIR / "per_residue_benchmarks"
    )
    _, tm_labels = load_tmbed_annotated(TMBED_FASTA)

    cb513_avail = sorted(set(emb_cb513.keys()) & set(ss3_labels.keys()))
    rng = random.Random(42)
    rng.shuffle(cb513_avail)
    n_tr = int(len(cb513_avail) * 0.8)
    cb_train, cb_test = cb513_avail[:n_tr], cb513_avail[n_tr:]

    tm_avail = sorted(set(emb_chezod.keys()) & set(tm_labels.keys()))
    rng2 = random.Random(42)
    rng2.shuffle(tm_avail)
    n_tr_tm = int(len(tm_avail) * 0.8)
    tm_train, tm_test = tm_avail[:n_tr_tm], tm_avail[n_tr_tm:]

    metadata = load_metadata_csv(META_5K)
    metadata, kept_ids = filter_by_family_size(metadata, min_members=3)
    with open(SPLIT_5K) as f:
        split = json.load(f)
    ret_test_ids = [i for i in split["test_ids"] if i in kept_ids]

    # Compute full PCA on 5K corpus
    print("  Computing corpus PCA (full rotation matrix)...")
    stats = compute_corpus_stats(emb_5k, n_sample=50_000, n_pcs=5, seed=42)
    rotation = stats["rotation_matrix"]  # (D, D)
    mean_vec = stats["mean_vec"]         # (D,)

    step_results = {"pca": {}}

    for d_out in DIMS:
        print(f"\n  PCA d={d_out}...")
        t0 = time.time()

        # PCA project: center + rotate + truncate
        cb_proj = project_pca(emb_cb513, rotation, mean_vec, d_out)
        cz_proj = project_pca(emb_chezod, rotation, mean_vec, d_out)
        fk_proj = project_pca(emb_5k, rotation, mean_vec, d_out)

        # Per-residue
        pr = evaluate_all_per_residue(
            cb_proj, cz_proj, cz_proj,
            ss3_labels, ss8_labels,
            disorder_scores, disorder_train, disorder_test,
            tm_labels, tm_train, tm_test,
            cb_train, cb_test,
        )

        # Retrieval (using DCT K=4 protein vector)
        ret = evaluate_retrieval(fk_proj, metadata, ret_test_ids, d_out)

        elapsed = time.time() - t0
        step_results["pca"][str(d_out)] = {**pr, **ret, "elapsed_s": round(elapsed, 1)}
        print(f"    PCA-{d_out}: SS3={pr['ss3_q3']:.3f} SS8={pr['ss8_q8']:.3f} "
              f"Dis={pr['disorder_rho']:.3f} TM={pr['tm_f1']:.3f} "
              f"Ret@1={ret['family_ret1']:.3f} ({elapsed:.1f}s)")

    results["B"] = step_results
    results["steps_done"].append("B")
    save_results(results)
    print("\n  Step B complete!")
    return results
```

- [ ] **Step 2: Run Step B**

Run: `uv run python experiments/38_projection_quality_diagnostic.py`
Expected: PCA results at 512/640/768 for comparison with Step A

- [ ] **Step 3: Commit**

```bash
git add experiments/38_projection_quality_diagnostic.py
git commit -m "feat(exp38): step B — PCA dimension sweep"
```

---

### Task 4: Step C — Channel importance analysis + main block

**Files:**
- Modify: `experiments/38_projection_quality_diagnostic.py`

- [ ] **Step 1: Implement step_C and main block**

```python
def step_C(results):
    """Channel importance: which raw 1024d channels matter for TM and disorder?"""
    if "C" in results.get("steps_done", []):
        print("Step C already done, skipping")
        return results

    print("\n" + "=" * 60)
    print("Step C: Channel Importance Analysis")
    print("=" * 60)

    # Load embeddings + labels
    emb_chezod = {}
    if EMB_CHEZOD.exists():
        emb_chezod = load_residue_embeddings(EMB_CHEZOD)
    if EMB_VALID.exists():
        emb_valid = load_residue_embeddings(EMB_VALID)
        for k, v in emb_valid.items():
            if k.startswith("chezod_") or k.startswith("tmbed_"):
                emb_chezod[k] = v

    emb_cb513 = load_residue_embeddings(EMB_CB513)
    _, ss3_labels, _, _ = load_cb513_csv(CB513_CSV)
    _, disorder_scores, disorder_train, disorder_test = load_chezod_seth(
        DATA_DIR / "per_residue_benchmarks"
    )
    _, tm_labels = load_tmbed_annotated(TMBED_FASTA)

    # Compute per-channel correlation with disorder scores
    print("  Computing per-channel disorder correlation...")
    dis_proteins = sorted(set(emb_chezod.keys()) & set(disorder_scores.keys()))
    all_emb_rows = []
    all_dis_scores = []
    for pid in dis_proteins:
        emb = emb_chezod[pid].astype(np.float32)
        scores = disorder_scores[pid]
        L = min(len(scores), emb.shape[0])
        all_emb_rows.append(emb[:L])
        all_dis_scores.append(scores[:L])

    X_dis = np.vstack(all_emb_rows)  # (N_residues, 1024)
    y_dis = np.concatenate(all_dis_scores)  # (N_residues,)
    D = X_dis.shape[1]

    from scipy.stats import spearmanr
    dis_channel_corr = np.zeros(D)
    for ch in range(D):
        rho, _ = spearmanr(X_dis[:, ch], y_dis)
        dis_channel_corr[ch] = abs(rho) if not np.isnan(rho) else 0.0

    # Compute per-channel F-stat for TM topology (multi-class)
    print("  Computing per-channel TM importance...")
    tm_avail = sorted(set(emb_chezod.keys()) & set(tm_labels.keys()))
    all_tm_emb = []
    all_tm_labels = []
    TM_MAP = {"H": 0, "B": 1, "S": 2, "O": 3, "h": 0, "b": 1, "s": 2, "o": 3}
    for pid in tm_avail:
        emb = emb_chezod[pid].astype(np.float32)
        labels_str = tm_labels[pid]
        L = min(len(labels_str), emb.shape[0])
        mapped = []
        for c in labels_str[:L]:
            if c in TM_MAP:
                mapped.append(TM_MAP[c])
            else:
                mapped.append(3)  # default to 'other'
        all_tm_emb.append(emb[:L])
        all_tm_labels.append(np.array(mapped))

    if all_tm_emb:
        X_tm = np.vstack(all_tm_emb)
        y_tm = np.concatenate(all_tm_labels)

        # Per-channel F-statistic (ANOVA)
        from sklearn.feature_selection import f_classif
        f_scores, p_vals = f_classif(X_tm, y_tm)
        tm_channel_f = f_scores
    else:
        tm_channel_f = np.zeros(D)
        print("  WARNING: No TM proteins found!")

    # Compute PCA stats — where does important info concentrate?
    print("  Computing PCA concentration of task-important channels...")
    stats = compute_corpus_stats(
        load_residue_embeddings(EMB_5K), n_sample=50_000, n_pcs=5, seed=42
    )
    rotation = stats["rotation_matrix"]  # (D, D) — rows are PCs

    # For each PC direction, compute weighted importance for disorder and TM
    # PC_importance[i] = sum_j |rotation[i,j]| * channel_importance[j]
    dis_pc_importance = np.abs(rotation) @ dis_channel_corr  # (D,)
    tm_pc_importance = np.abs(rotation) @ tm_channel_f       # (D,)

    # Cumulative importance captured by top-k PCs
    dis_cumulative = np.cumsum(dis_pc_importance) / dis_pc_importance.sum()
    tm_cumulative = np.cumsum(tm_pc_importance) / tm_pc_importance.sum()

    step_results = {
        "disorder_top10_channels": int(np.argsort(dis_channel_corr)[-10:][::-1].tolist()[0]),
        "disorder_channel_corr_stats": {
            "max": float(dis_channel_corr.max()),
            "mean": float(dis_channel_corr.mean()),
            "std": float(dis_channel_corr.std()),
            "top20_channels": np.argsort(dis_channel_corr)[-20:][::-1].tolist(),
            "top20_values": dis_channel_corr[np.argsort(dis_channel_corr)[-20:][::-1]].tolist(),
        },
        "tm_channel_f_stats": {
            "max": float(tm_channel_f.max()),
            "mean": float(tm_channel_f.mean()),
            "std": float(tm_channel_f.std()),
            "top20_channels": np.argsort(tm_channel_f)[-20:][::-1].tolist(),
            "top20_values": tm_channel_f[np.argsort(tm_channel_f)[-20:][::-1]].tolist(),
        },
        "pca_cumulative_disorder": {
            f"top_{k}": float(dis_cumulative[k - 1]) for k in [64, 128, 256, 384, 512, 640, 768]
            if k <= len(dis_cumulative)
        },
        "pca_cumulative_tm": {
            f"top_{k}": float(tm_cumulative[k - 1]) for k in [64, 128, 256, 384, 512, 640, 768]
            if k <= len(tm_cumulative)
        },
        "n_residues_disorder": int(X_dis.shape[0]),
        "n_residues_tm": int(X_tm.shape[0]) if all_tm_emb else 0,
    }

    results["C"] = step_results
    results["steps_done"].append("C")
    save_results(results)
    print("\n  Step C complete!")
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results):
    """Print a comparison table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY: Projection Quality Diagnostic")
    print("=" * 80)

    if "A" in results:
        raw = results["A"]["raw_1024d"]
        print(f"\n{'Method':<20} {'SS3 Q3':>8} {'SS8 Q8':>8} {'Dis rho':>8} {'TM F1':>8} {'Ret@1':>8}")
        print("-" * 72)
        print(f"{'Raw 1024d':<20} {raw['ss3_q3']:>8.3f} {raw['ss8_q8']:>8.3f} "
              f"{raw['disorder_rho']:>8.3f} {raw['tm_f1']:>8.3f} {'—':>8}")

        for d in DIMS:
            ds = str(d)
            if ds in results["A"]["rp"]:
                r = results["A"]["rp"][ds]
                print(f"{'RP ' + ds:<20} {r['ss3_q3']:>8.3f} {r['ss8_q8']:>8.3f} "
                      f"{r['disorder_rho']:>8.3f} {r['tm_f1']:>8.3f} {r['family_ret1']:>8.3f}")

    if "B" in results:
        for d in DIMS:
            ds = str(d)
            if ds in results["B"]["pca"]:
                r = results["B"]["pca"][ds]
                print(f"{'PCA ' + ds:<20} {r['ss3_q3']:>8.3f} {r['ss8_q8']:>8.3f} "
                      f"{r['disorder_rho']:>8.3f} {r['tm_f1']:>8.3f} {r['family_ret1']:>8.3f}")

    if "C" in results:
        print(f"\n  PCA cumulative disorder importance:")
        for k, v in results["C"]["pca_cumulative_disorder"].items():
            print(f"    {k}: {v:.3f}")
        print(f"\n  PCA cumulative TM importance:")
        for k, v in results["C"]["pca_cumulative_tm"].items():
            print(f"    {k}: {v:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Experiment 38: Projection Quality Diagnostic")
    print("=" * 60)

    results = load_results()

    results = step_A(results)
    results = step_B(results)
    results = step_C(results)

    print_summary(results)
    save_results(results)
    print(f"\nResults saved to {RESULTS_PATH}")
```

- [ ] **Step 2: Run full experiment**

Run: `uv run python experiments/38_projection_quality_diagnostic.py`
Expected: Complete results with comparison table showing RP vs PCA at each dimension

- [ ] **Step 3: Commit results**

```bash
git add experiments/38_projection_quality_diagnostic.py data/benchmarks/projection_diagnostic_results.json
git commit -m "data(exp38): projection quality diagnostic — RP vs PCA dimension sweep + channel importance"
```

---

### Task 5: Analyze results and document findings

**Files:**
- Read: `data/benchmarks/projection_diagnostic_results.json`

- [ ] **Step 1: Review results and create summary**

After running the experiment, analyze:
1. Does PCA 512d close the TM F1 gap compared to RP 512d?
2. Does RP 768d close the gap?
3. How much task-important information concentrates in PCA top-512 vs top-768?
4. Is the retrieval difference between PCA and RP still within noise at each dimension?

- [ ] **Step 2: Update memory with findings**

Update `project_perresidue_quality_gap.md` with the experiment results.

- [ ] **Step 3: Commit documentation**

```bash
git add -A
git commit -m "docs(exp38): analysis of projection quality diagnostic results"
```
