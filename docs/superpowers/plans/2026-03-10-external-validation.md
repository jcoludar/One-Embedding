# External Validation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that compressed ProtT5 d256 embeddings preserve biological signal on external tasks (ToxFam binary toxicity classification) and fill the TMbed PCA gap.

**Architecture:** A single experiment script (`experiments/15_external_validation.py`) handles subset selection, streaming extraction+compression, and H5 generation. ToxFam's existing CLI pipeline (`toxfam train`) runs the binary classification. A secondary step runs PCA-256 on TMbed for comparison with ChannelCompressor.

**Tech Stack:** PyTorch, h5py, pandas, ProtT5 (cached), ChannelCompressor checkpoint, ToxFam CLI

**Disk budget:** ~40 MB new files total. Per-residue embeddings are NEVER saved to disk — streamed through compression in memory.

---

## Chunk 1: Core Pipeline

### Task 1: Subset Selection Script

**Files:**
- Create: `experiments/15_external_validation.py`

The script reads ToxFam's training CSV, selects all toxic proteins plus a balanced random sample of non-toxic proteins (stratified by split to preserve train/val/test ratios), and writes a subset CSV.

- [ ] **Step 1: Create experiment script with subset selection (step S1)**

```python
"""
Experiment 15: External Validation
===================================
Validates compressed ProtT5 d256 embeddings on:
  A) ToxFam binary toxicity classification (balanced 7K subset)
  B) TMbed PCA-256 baseline comparison

Usage:
  uv run python experiments/15_external_validation.py --step S1   # Select balanced subset
  uv run python experiments/15_external_validation.py --step S2   # Extract + compress embeddings
  uv run python experiments/15_external_validation.py --step S3   # TMbed PCA baseline
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOXFAM_ROOT = Path("/Users/jcoludar/CascadeProjects/students/ToxFam")
TOXFAM_CSV = TOXFAM_ROOT / "data" / "processed" / "training_data.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "external_validation" / "toxfam"
RESULTS_FILE = PROJECT_ROOT / "data" / "benchmarks" / "external_validation_results.json"

# ToxFam label conventions
NONTOXIN_LABELS = {"nontox", "nontoxic"}


def is_toxic(label: str) -> bool:
    return str(label).strip().lower() not in NONTOXIN_LABELS


def step_s1_select_subset(seed: int = 42):
    """Select balanced subset: all toxic + matched non-toxic, stratified by split."""
    print("=" * 60)
    print("Step S1: Select balanced ToxFam subset")
    print("=" * 60)

    df = pd.read_csv(TOXFAM_CSV)
    print(f"Total proteins: {len(df)}")

    df["is_toxic"] = df["Protein families"].apply(is_toxic)
    toxic = df[df["is_toxic"]]
    nontoxic = df[~df["is_toxic"]]
    print(f"Toxic: {len(toxic)}, Non-toxic: {len(nontoxic)}")

    # Sample non-toxic to match toxic count, stratified by split
    rng = np.random.RandomState(seed)
    sampled_parts = []
    for split_name in ["train", "val", "test"]:
        nt_split = nontoxic[nontoxic["Split"] == split_name]
        t_split = toxic[toxic["Split"] == split_name]
        n_sample = min(len(t_split), len(nt_split))
        sampled = nt_split.sample(n=n_sample, random_state=rng)
        sampled_parts.append(sampled)
        print(f"  {split_name}: {len(t_split)} toxic + {n_sample} non-toxic sampled")

    nontoxic_sampled = pd.concat(sampled_parts)
    subset = pd.concat([toxic, nontoxic_sampled]).sort_index()
    subset["binary_label"] = subset["is_toxic"].map({True: "toxin", False: "nontoxin"})

    print(f"\nSubset total: {len(subset)}")
    for split_name in ["train", "val", "test"]:
        s = subset[subset["Split"] == split_name]
        print(f"  {split_name}: {len(s)} ({s['is_toxic'].sum()} toxic, "
              f"{(~s['is_toxic']).sum()} non-toxic)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subset_path = OUTPUT_DIR / "subset.csv"
    subset.to_csv(subset_path, index=False)
    print(f"\nSaved subset CSV: {subset_path}")
    print(f"Columns: {list(subset.columns)}")
    return subset


def main():
    parser = argparse.ArgumentParser(description="Experiment 15: External Validation")
    parser.add_argument("--step", required=True, choices=["S1", "S2", "S3"],
                        help="Which step to run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.step == "S1":
        step_s1_select_subset(seed=args.seed)
    elif args.step == "S2":
        step_s2_extract_and_compress(seed=args.seed)
    elif args.step == "S3":
        step_s3_tmbed_pca()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run step S1 to verify subset selection**

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run python experiments/15_external_validation.py --step S1`

Expected: Prints counts showing ~3.4K toxic + ~3.4K non-toxic, stratified by split. Creates `data/external_validation/toxfam/subset.csv`.

- [ ] **Step 3: Commit subset selection**

```bash
git add experiments/15_external_validation.py
git commit -m "feat(exp15): add subset selection for ToxFam external validation"
```

---

### Task 2: Streaming Extraction + Compression

**Files:**
- Modify: `experiments/15_external_validation.py`

Add step S2: loads ProtT5 (once) and ChannelCompressor, extracts all per-residue embeddings in a single call, then compresses. For each protein, computes mean-pooled 1024d baseline AND compressed mean-pooled 256d. Saves both as H5 files in ToxFam format (key=identifier, value=shape (D,), float32). Per-residue embeddings are never saved to disk.

**Note on truncation**: ProtT5 tokenizer truncates at 512 tokens. Proteins >510 AA will have shorter embeddings. Both baseline mean-pool and compressed mean-pool operate on the same truncated sequence, so the comparison is fair.

- [ ] **Step 1: Add step S2 to the experiment script**

Insert before `main()`:

```python
def step_s2_extract_and_compress(seed: int = 42):
    """Extract ProtT5 per-residue embeddings, compress, save mean-pooled H5 files."""
    import shutil
    print("=" * 60)
    print("Step S2: Streaming extraction + compression")
    print("=" * 60)

    # Disk space safety check
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    print(f"Free disk: {free_gb:.1f} GB")
    if free_gb < 1.0:
        print("ERROR: Less than 1 GB free, aborting")
        return

    from src.extraction.prot_t5_extractor import extract_prot_t5_embeddings
    from src.compressors.channel_compressor import ChannelCompressor
    from src.utils.device import get_device

    device = get_device()
    print(f"Device: {device}")

    # Load subset
    subset_path = OUTPUT_DIR / "subset.csv"
    if not subset_path.exists():
        print("Subset CSV not found — running S1 first")
        step_s1_select_subset(seed=seed)
    subset = pd.read_csv(subset_path)
    print(f"Subset: {len(subset)} proteins")

    # Load ChannelCompressor checkpoint
    ckpt_path = (PROJECT_ROOT / "data" / "checkpoints" / "channel" /
                 "channel_prot_t5_contrastive_d256_s42" / "best_model.pt")
    model = ChannelCompressor(input_dim=1024, latent_dim=256, dropout=0.1,
                              use_residual=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    model = model.to(device).eval()
    print(f"Loaded ChannelCompressor from {ckpt_path}")

    # Build fasta dict from CSV sequences
    fasta_dict = dict(zip(subset["identifier"], subset["Sequence"]))
    print(f"Sequences to extract: {len(fasta_dict)}")

    # Extract ALL per-residue ProtT5 embeddings in one call
    # (avoids reloading the 21 GB model per chunk — internal batching handles memory)
    print(f"\nExtracting ProtT5 per-residue embeddings for {len(fasta_dict)} proteins...")
    print("This takes ~20-30 min. Monitor heat: pmset -g therm")
    per_residue = extract_prot_t5_embeddings(
        fasta_dict,
        model_name="Rostlab/prot_t5_xl_uniref50",
        batch_size=4,
        device=device,
    )
    print(f"Extracted {len(per_residue)} protein embeddings")

    # Compress and save — iterate once, write both H5 files
    baseline_h5_path = OUTPUT_DIR / "embeddings_baseline_1024.h5"
    compressed_h5_path = OUTPUT_DIR / "embeddings_compressed_256.h5"

    with h5py.File(baseline_h5_path, "w") as h5_base, \
         h5py.File(compressed_h5_path, "w") as h5_comp:

        with torch.no_grad():
            for i, (pid, emb) in enumerate(per_residue.items()):
                # emb shape: (L, 1024) — may be <seq_len due to 512-token truncation
                # Baseline: mean-pool to 1024d
                baseline_vec = emb.mean(axis=0)  # (1024,)
                h5_base.create_dataset(pid, data=baseline_vec.astype(np.float32))

                # Compressed: run through ChannelCompressor
                L = emb.shape[0]
                states = torch.from_numpy(emb).unsqueeze(0).to(device)  # (1, L, 1024)
                mask = torch.ones(1, L, device=device)
                latent = model.compress(states, mask)  # (1, L, 256)
                pooled = model.get_pooled(latent, strategy="mean")  # (1, 256)
                comp_vec = pooled[0].cpu().numpy()  # (256,)
                h5_comp.create_dataset(pid, data=comp_vec.astype(np.float32))

                if (i + 1) % 1000 == 0:
                    print(f"  Compressed {i + 1}/{len(per_residue)}")
                    if device.type == "mps":
                        torch.mps.empty_cache()

    # Free per-residue embeddings from memory
    del per_residue
    if device.type == "mps":
        torch.mps.empty_cache()

    base_size = baseline_h5_path.stat().st_size / 1e6
    comp_size = compressed_h5_path.stat().st_size / 1e6
    print(f"\nBaseline H5: {baseline_h5_path} ({base_size:.1f} MB)")
    print(f"Compressed H5: {compressed_h5_path} ({comp_size:.1f} MB)")
    print("Done. No per-residue embeddings saved to disk.")
```

- [ ] **Step 2: Run step S2**

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run python experiments/15_external_validation.py --step S2`

Expected: ~25-30 min. Prints chunk progress. Creates two H5 files totaling ~35 MB in `data/external_validation/toxfam/`. Monitor system heat with `pmset -g therm` periodically.

- [ ] **Step 3: Verify H5 files match expected format**

Run quick validation:
```bash
cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run python -c "
import h5py
for name, path in [('baseline', 'data/external_validation/toxfam/embeddings_baseline_1024.h5'),
                   ('compressed', 'data/external_validation/toxfam/embeddings_compressed_256.h5')]:
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        print(f'{name}: {len(keys)} proteins, shape={f[keys[0]].shape}, dtype={f[keys[0]].dtype}')
"
```

Expected:
```
baseline: ~6800 proteins, shape=(1024,), dtype=float32
compressed: ~6800 proteins, shape=(256,), dtype=float32
```

- [ ] **Step 4: Commit extraction pipeline**

```bash
git add experiments/15_external_validation.py
git commit -m "feat(exp15): add streaming ProtT5 extraction + compression for ToxFam"
```

---

### Task 3: ToxFam Integration — Config + Run

**Files:**
- Create: `configs/compressed_baseline_1024.yaml` (in ToxFam project)
- Create: `configs/compressed_256.yaml` (in ToxFam project)

Create two ToxFam config YAMLs and a subset CSV in ToxFam format, then run ToxFam's binary training pipeline on both.

- [ ] **Step 1: Create subset CSV in ToxFam format**

The subset CSV from step S1 has extra columns (`is_toxic`, `binary_label`). ToxFam expects exactly: `identifier`, `Sequence`, `Protein families`, `Organism (ID)`, `Split`. Copy the subset CSV with only the required columns:

```bash
cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run python -c "
import pandas as pd
from pathlib import Path

subset = pd.read_csv('data/external_validation/toxfam/subset.csv')
toxfam_csv = subset[['identifier', 'Sequence', 'Protein families', 'Organism (ID)', 'Split']]
out_path = Path('/Users/jcoludar/CascadeProjects/students/ToxFam/data/processed/subset_balanced.csv')
toxfam_csv.to_csv(out_path, index=False)
print(f'Saved {len(toxfam_csv)} rows to {out_path}')
"
```

- [ ] **Step 2: Symlink H5 files into ToxFam data directory**

```bash
ln -sf /Users/jcoludar/CascadeProjects/ProteEmbedExplorations/data/external_validation/toxfam/embeddings_baseline_1024.h5 \
       /Users/jcoludar/CascadeProjects/students/ToxFam/data/processed/embeddings_baseline_1024.h5

ln -sf /Users/jcoludar/CascadeProjects/ProteEmbedExplorations/data/external_validation/toxfam/embeddings_compressed_256.h5 \
       /Users/jcoludar/CascadeProjects/students/ToxFam/data/processed/embeddings_compressed_256.h5
```

- [ ] **Step 3: Create ToxFam config for 1024d baseline**

Create `/Users/jcoludar/CascadeProjects/students/ToxFam/configs/compressed_baseline_1024.yaml`:

```yaml
# Baseline: ProtT5 mean-pooled 1024d on balanced subset
# Uses same ProtT5 model as ChannelCompressor training (prot_t5_xl_uniref50)
input_csv: "data/processed/subset_balanced.csv"
h5_paths_glob: "data/processed/embeddings_baseline_1024.h5"
training_strategy: "binary"
embedding_dim: 1024
hidden_dims: [256, 256]
dropout: 0.5
batch_size: 64
num_epochs: 200
learning_rate: 0.0001
early_stopping_patience: 10
output_dir: "model/model_output/compressed_baseline_1024_run"
```

- [ ] **Step 4: Create ToxFam config for 256d compressed**

Create `/Users/jcoludar/CascadeProjects/students/ToxFam/configs/compressed_256.yaml`:

```yaml
# Compressed: ChannelCompressor d256 contrastive on balanced subset
# Same proteins, same ProtT5 model, 4x compression
input_csv: "data/processed/subset_balanced.csv"
h5_paths_glob: "data/processed/embeddings_compressed_256.h5"
training_strategy: "binary"
embedding_dim: 256
hidden_dims: [256, 256]
dropout: 0.5
batch_size: 64
num_epochs: 200
learning_rate: 0.0001
early_stopping_patience: 10
output_dir: "model/model_output/compressed_256_run"
```

- [ ] **Step 5: Run ToxFam binary pipeline — 1024d baseline**

```bash
cd /Users/jcoludar/CascadeProjects/students/ToxFam
uv run toxfam train configs/compressed_baseline_1024.yaml
```

Expected: Trains binary MLP, early stops around epoch 30-50. Prints test metrics (ROC-AUC, PR-AUC, F1, MCC). Results saved to `model/model_output/compressed_baseline_1024_run/metrics/`.

- [ ] **Step 6: Run ToxFam binary pipeline — 256d compressed**

```bash
cd /Users/jcoludar/CascadeProjects/students/ToxFam
uv run toxfam train configs/compressed_256.yaml
```

Expected: Similar convergence. The key comparison is whether 256d compressed achieves comparable F1/PR-AUC to 1024d baseline.

- [ ] **Step 7: Compare results and record**

Read both metric files and compare:
```bash
cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run python -c "
import json
from pathlib import Path

toxfam = Path('/Users/jcoludar/CascadeProjects/students/ToxFam/model/model_output')

results = {}
for name, dirname in [('baseline_1024', 'compressed_baseline_1024_run'),
                       ('compressed_256', 'compressed_256_run')]:
    metrics_file = toxfam / dirname / 'metrics' / 'binary_test_calibrated_metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as fh:
            results[name] = json.load(fh)
    else:
        print(f'WARNING: {metrics_file} not found')

for name, m in results.items():
    print(f'{name}:')
    for k in ['roc_auc', 'pr_auc', 'f1', 'mcc', 'accuracy']:
        if k in m:
            print(f'  {k}: {m[k]:.4f}')
    print()
"
```

- [ ] **Step 8: Commit ToxFam configs**

```bash
cd /Users/jcoludar/CascadeProjects/students/ToxFam
git add configs/compressed_baseline_1024.yaml configs/compressed_256.yaml data/processed/subset_balanced.csv
git commit -m "feat: add compressed embedding validation configs for ProteEmbedExplorations"
```

---

## Chunk 2: TMbed PCA Baseline + Results Synthesis

### Task 4: TMbed PCA-256 Baseline

**Files:**
- Modify: `experiments/15_external_validation.py`

Add step S3: loads TMbed per-residue embeddings (already extracted in experiment 13), applies PCA-256, and runs the same linear probe topology evaluation. Compares PCA-256 with ChannelCompressor d256 on TMbed.

- [ ] **Step 1: Add step S3 to the experiment script**

Insert before `main()`:

```python
def step_s3_tmbed_pca():
    """Compare PCA-256 with ChannelCompressor on TMbed topology prediction."""
    print("=" * 60)
    print("Step S3: TMbed PCA-256 baseline comparison")
    print("=" * 60)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.preprocessing import LabelEncoder

    # --- Load TMbed data ---
    tmbed_fasta = PROJECT_ROOT / "data" / "per_residue_benchmarks" / "TMbed" / "cv_00_annotated.fasta"
    if not tmbed_fasta.exists():
        print(f"TMbed data not found at {tmbed_fasta}")
        return

    # Parse TMbed FASTA (3-line format: header, sequence, topology)
    proteins = []
    with open(tmbed_fasta) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            pid = lines[i][1:].split()[0]
            seq = lines[i + 1]
            topo = lines[i + 2]
            proteins.append((pid, seq, topo))
            i += 3
        else:
            i += 1

    print(f"TMbed proteins: {len(proteins)}")

    # Normalize topology labels: H/h->H, B/b->B, S->S, everything else->O
    def normalize_topo(char):
        if char in "Hh":
            return "H"
        elif char in "Bb":
            return "B"
        elif char == "S":
            return "S"
        else:
            return "O"

    # --- Load per-residue embeddings ---
    # Try both ESM2 and ProtT5 validation H5 files
    results = {}

    for plm_name, h5_name, input_dim in [
        ("ESM2-650M", "esm2_650m_validation.h5", 1280),
        ("ProtT5-XL", "prot_t5_xl_validation.h5", 1024),
    ]:
        h5_path = PROJECT_ROOT / "data" / "residue_embeddings" / h5_name
        if not h5_path.exists():
            print(f"  {plm_name} validation H5 not found, skipping")
            continue

        print(f"\n--- {plm_name} (dim={input_dim}) ---")

        # Collect all residue embeddings + labels
        all_embeddings = []
        all_labels = []

        with h5py.File(h5_path, "r") as h5f:
            matched = 0
            for pid, seq, topo in proteins:
                if pid not in h5f:
                    continue
                emb = h5f[pid][:]  # (L, D)
                L = min(len(topo), emb.shape[0])
                for j in range(L):
                    all_embeddings.append(emb[j])
                    all_labels.append(normalize_topo(topo[j]))
                matched += 1

        print(f"  Matched proteins: {matched}/{len(proteins)}")
        if matched == 0:
            continue

        X = np.array(all_embeddings, dtype=np.float32)
        y_str = np.array(all_labels)
        le = LabelEncoder()
        y = le.fit_transform(y_str)
        print(f"  Total residues: {len(y)}")
        print(f"  Classes: {dict(zip(le.classes_, np.bincount(y)))}")

        # --- Evaluate: Original, PCA-256, PCA-128 ---
        seeds = [42, 123, 456]
        for method_name, X_method in [
            (f"original_{input_dim}", X),
            ("PCA-256", None),
            ("PCA-128", None),
        ]:
            dim = int(method_name.split("-")[1]) if "PCA" in method_name else input_dim

            accs, f1s = [], []
            for s in seeds:
                rng = np.random.RandomState(s)
                idx = rng.permutation(len(y))
                split = int(0.8 * len(idx))
                train_idx, test_idx = idx[:split], idx[split:]

                if "PCA" in method_name:
                    pca = PCA(n_components=dim, random_state=s)
                    X_train = pca.fit_transform(X[train_idx])
                    X_test = pca.transform(X[test_idx])
                else:
                    X_train = X_method[train_idx]
                    X_test = X_method[test_idx]

                clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                         random_state=s)
                clf.fit(X_train, y[train_idx])
                y_pred = clf.predict(X_test)
                accs.append(accuracy_score(y[test_idx], y_pred))
                f1s.append(f1_score(y[test_idx], y_pred, average="macro"))

            acc_mean, acc_std = np.mean(accs), np.std(accs)
            f1_mean, f1_std = np.mean(f1s), np.std(f1s)
            print(f"  {method_name}: Acc={acc_mean:.4f}±{acc_std:.4f}, "
                  f"F1={f1_mean:.4f}±{f1_std:.4f}")

            results[f"{plm_name}_{method_name}"] = {
                "accuracy_mean": round(float(acc_mean), 4),
                "accuracy_std": round(float(acc_std), 4),
                "f1_mean": round(float(f1_mean), 4),
                "f1_std": round(float(f1_std), 4),
            }

    # Load ChannelCompressor TMbed results from experiment 13 for comparison
    robust_results_path = PROJECT_ROOT / "data" / "benchmarks" / "robust_validation_results.json"
    if robust_results_path.exists():
        with open(robust_results_path) as f:
            robust = json.load(f)
        print("\n--- Comparison with ChannelCompressor d256 (from Experiment 13) ---")
        # Extract TMbed entries from robust validation results
        for key, val in robust.items():
            if "tmbed" in str(key).lower() or "TMbed" in str(key):
                print(f"  {key}: {val}")

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
    existing["tmbed_pca_comparison"] = results
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
```

Also update `main()` to wire up S3:

```python
    elif args.step == "S3":
        step_s3_tmbed_pca()
```

- [ ] **Step 2: Run step S3**

Run: `cd /Users/jcoludar/CascadeProjects/ProteEmbedExplorations && uv run python experiments/15_external_validation.py --step S3`

Expected: Runs in ~2-5 minutes (no GPU needed, just PCA + logistic regression on pre-extracted embeddings). Prints comparison table showing original vs PCA-256 vs PCA-128 for both PLMs.

- [ ] **Step 3: Commit TMbed PCA baseline**

```bash
git add experiments/15_external_validation.py
git commit -m "feat(exp15): add TMbed PCA-256 baseline comparison"
```

---

### Task 5: Record Results in ANALYSIS.md

**Files:**
- Modify: `ANALYSIS.md`

- [ ] **Step 1: Add ToxFam results section to ANALYSIS.md**

After the ToxFam binary pipeline completes (Task 3, Steps 5-7), add a new subsection under Section 7 ("The One Embedding Question") with the ToxFam binary classification comparison table:

```markdown
### External Validation: ToxFam Binary Toxicity (Experiment 15)

Balanced subset (~3.4K toxic + ~3.4K non-toxic) from ToxFam dataset, identity-aware splits at 30% identity. Same ProtT5 model (prot_t5_xl_uniref50) for both baseline and compressed.

| Embedding | Dim | ROC-AUC | PR-AUC | F1 | MCC |
|-----------|:---:|:-------:|:------:|:--:|:---:|
| ProtT5 mean-pool (baseline) | 1024 | X.XXX | X.XXX | X.XXX | X.XXX |
| ChannelCompressor d256 | 256 | X.XXX | X.XXX | X.XXX | X.XXX |
| Retention | — | XX.X% | XX.X% | XX.X% | XX.X% |
```

- [ ] **Step 2: Add TMbed PCA comparison to ANALYSIS.md**

Add under the cross-dataset validation section:

```markdown
### TMbed PCA Baseline (Experiment 15)

| Method | Dim | Macro F1 | vs Original |
|--------|:---:|:--------:|:-----------:|
| ESM2 original | 1280 | 0.865 | 100% |
| ESM2 PCA-256 | 256 | X.XXX | XX.X% |
| ESM2 ChannelComp d256 | 256 | 0.761 | 88.0% |
| ProtT5 original | 1024 | 0.795 | 100% |
| ProtT5 PCA-256 | 256 | X.XXX | XX.X% |
| ProtT5 ChannelComp d256 | 256 | 0.657 | 82.6% |
```

- [ ] **Step 3: Update memory file with results**

Update `/Users/jcoludar/.claude/projects/-Users-jcoludar-CascadeProjects-ProteEmbedExplorations/memory/MEMORY.md` with Phase 9C status and key findings.

- [ ] **Step 4: Final commit**

```bash
git add ANALYSIS.md experiments/15_external_validation.py data/benchmarks/external_validation_results.json
git commit -m "docs: add Phase 9C external validation results (ToxFam + TMbed PCA)"
```

---

## Execution Order

1. **Task 1** (S1 subset selection) — 1 min
2. **Task 2** (S2 extraction + compression) — 25-30 min, GPU-intensive
3. **Task 3** (ToxFam integration + training) — ~5 min per run, sequential
4. **Task 4** (S3 TMbed PCA) — 2-5 min, CPU only, can run while reviewing Task 3 results
5. **Task 5** (documentation) — 5 min

**Total: ~45 min**

## Disk Space Budget

| Item | Size | Cumulative |
|------|-----:|----------:|
| `subset.csv` | <1 MB | <1 MB |
| `embeddings_baseline_1024.h5` | ~28 MB | ~29 MB |
| `embeddings_compressed_256.h5` | ~7 MB | ~36 MB |
| `subset_balanced.csv` (ToxFam copy) | <1 MB | ~37 MB |
| ToxFam model outputs (2 runs) | ~3 MB | ~40 MB |
| **Total new disk** | | **~40 MB** |

No per-residue embeddings saved. ProtT5 model already cached (21 GB). Safe for 18 GB free.

## Success Criteria

- **ToxFam**: Compressed 256d achieves >= 90% of baseline 1024d on PR-AUC and F1
- **TMbed PCA**: If PCA-256 F1 is close to ChannelCompressor F1, the TMbed "loss" is inherent to dimensionality reduction, not our method
