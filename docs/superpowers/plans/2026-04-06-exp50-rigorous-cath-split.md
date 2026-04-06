# Exp 50 Rigorous CATH-Split Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run Exp 50 (sequence → binary One Embedding CNN) on CATH20 (14,433 proteins) with CATH-level cluster holdouts (homologous-superfamily and topology), multi-seed variance, MMseqs2 leakage audit, and train-only codec centering — replacing the naive random 80/10/10 sighting run.

**Architecture:** Add a new splits module (`src/one_embedding/seq2oe_splits.py`) that parses the CATH-labeled FASTA and does whole-cluster holdout splits with per-Class greedy stratification. Refactor `experiments/50_sequence_to_oe.py` to dispatch on `--dataset` and `--split`, fit the OneEmbeddingCodec on train embeddings only, and write results under `results/exp50_rigorous/{split}_split/stage{N}/seed{seed}/`. Add a standalone MMseqs2 audit script and an orchestration runner that loops 2 stages × 2 splits × 3 seeds and aggregates per-seed JSONs into a final comparison table.

**Tech Stack:** PyTorch (MPS), NumPy, h5py, MMseqs2 (already installed at `/opt/homebrew/bin/mmseqs`), existing `OneEmbeddingCodec` and `Seq2OE_CNN` unchanged.

**Spec reference:** `docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md`

---

### Task 1: CATH FASTA parser

**Files:**
- Create: `src/one_embedding/seq2oe_splits.py`
- Create: `tests/test_seq2oe_splits.py`

- [ ] **Step 1: Write failing parser tests**

```python
# tests/test_seq2oe_splits.py
"""Tests for Seq2OE CATH-level cluster splits."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.one_embedding.seq2oe_splits import (
    parse_cath_fasta,
    cath_cluster_split,
    save_split,
    load_split,
)


SAMPLE_FASTA = """>12asA00|3.30.930.10
MKTAYIAKQRQIS
>132lA00|1.10.530.10
XVFGRCELAAAM
>153lA00|1.10.530.10
RTDCYGNVNRIDT
>16pkA02|3.40.50.1260
YFAKVLGNPPRP
>16vpA00|1.10.1290.10
SRMPSPPMPVPP
>1914A00|3.30.720.10
MVLLESEQFL
"""


class TestParseCathFasta:
    def test_parses_ids_and_codes(self, tmp_path):
        f = tmp_path / "cath.fa"
        f.write_text(SAMPLE_FASTA)
        meta = parse_cath_fasta(f)
        assert set(meta.keys()) == {
            "12asA00", "132lA00", "153lA00",
            "16pkA02", "16vpA00", "1914A00",
        }
        assert meta["12asA00"]["seq"] == "MKTAYIAKQRQIS"
        assert meta["12asA00"]["C"] == 3
        assert meta["12asA00"]["A"] == 30
        assert meta["12asA00"]["T"] == "3.30.930"
        assert meta["12asA00"]["H"] == "3.30.930.10"

    def test_rejects_malformed_header(self, tmp_path):
        f = tmp_path / "bad.fa"
        f.write_text(">noCode\nAAAA\n")
        with pytest.raises(ValueError, match="no CATH code"):
            parse_cath_fasta(f)

    def test_rejects_malformed_code(self, tmp_path):
        f = tmp_path / "bad.fa"
        f.write_text(">pid|1.2.3\nAAAA\n")  # only 3 levels
        with pytest.raises(ValueError, match="expected 4 dot-separated"):
            parse_cath_fasta(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seq2oe_splits.py::TestParseCathFasta -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.one_embedding.seq2oe_splits'`

- [ ] **Step 3: Implement the parser**

```python
# src/one_embedding/seq2oe_splits.py
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
```

- [ ] **Step 4: Run tests to verify parser passes**

Run: `uv run pytest tests/test_seq2oe_splits.py::TestParseCathFasta -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/seq2oe_splits.py tests/test_seq2oe_splits.py
git commit -m "feat(exp50): CATH-labeled FASTA parser for rigorous splits"
```

---

### Task 2: CATH cluster split with per-Class greedy stratification

**Files:**
- Modify: `src/one_embedding/seq2oe_splits.py`
- Modify: `tests/test_seq2oe_splits.py`

- [ ] **Step 1: Write failing tests for cath_cluster_split**

Append to `tests/test_seq2oe_splits.py`:

```python
class TestCathClusterSplit:
    def _make_meta(self):
        """10 proteins in 2 classes, 6 H-codes, 4 T-codes."""
        return {
            # Class 1, T=1.10.530, H=1.10.530.10 (2 proteins)
            "p1": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.10"},
            "p2": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.10"},
            # Class 1, T=1.10.530, H=1.10.530.20 (2 proteins)
            "p3": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.20"},
            "p4": {"seq": "A", "C": 1, "A": 10, "T": "1.10.530", "H": "1.10.530.20"},
            # Class 1, T=1.10.290, H=1.10.290.10 (1 protein)
            "p5": {"seq": "A", "C": 1, "A": 10, "T": "1.10.290", "H": "1.10.290.10"},
            # Class 3, T=3.30.930, H=3.30.930.10 (3 proteins)
            "p6": {"seq": "A", "C": 3, "A": 30, "T": "3.30.930", "H": "3.30.930.10"},
            "p7": {"seq": "A", "C": 3, "A": 30, "T": "3.30.930", "H": "3.30.930.10"},
            "p8": {"seq": "A", "C": 3, "A": 30, "T": "3.30.930", "H": "3.30.930.10"},
            # Class 3, T=3.30.720, H=3.30.720.10 (2 proteins)
            "p9": {"seq": "A", "C": 3, "A": 30, "T": "3.30.720", "H": "3.30.720.10"},
            "p10": {"seq": "A", "C": 3, "A": 30, "T": "3.30.720", "H": "3.30.720.10"},
        }

    def test_h_split_no_cluster_leakage(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        train_h = {meta[p]["H"] for p in train}
        val_h = {meta[p]["H"] for p in val}
        test_h = {meta[p]["H"] for p in test}
        assert train_h.isdisjoint(val_h)
        assert train_h.isdisjoint(test_h)
        assert val_h.isdisjoint(test_h)

    def test_t_split_no_cluster_leakage(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="T", fractions=(0.6, 0.2, 0.2), seed=42
        )
        train_t = {meta[p]["T"] for p in train}
        val_t = {meta[p]["T"] for p in val}
        test_t = {meta[p]["T"] for p in test}
        assert train_t.isdisjoint(val_t)
        assert train_t.isdisjoint(test_t)
        assert val_t.isdisjoint(test_t)

    def test_every_protein_assigned_exactly_once(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        all_assigned = set(train) | set(val) | set(test)
        assert all_assigned == set(meta.keys())
        assert len(train) + len(val) + len(test) == len(meta)

    def test_deterministic_same_seed(self):
        meta = self._make_meta()
        r1 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        r2 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        meta = self._make_meta()
        r1 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42)
        r2 = cath_cluster_split(meta, level="H", fractions=(0.6, 0.2, 0.2), seed=7)
        # With 6 H-codes and small counts they may occasionally collide;
        # assert at least one fold differs
        assert r1 != r2

    def test_invalid_level_raises(self):
        meta = self._make_meta()
        with pytest.raises(ValueError, match="level must be"):
            cath_cluster_split(meta, level="X", fractions=(0.6, 0.2, 0.2), seed=42)

    def test_fractions_must_sum_to_one(self):
        meta = self._make_meta()
        with pytest.raises(ValueError, match="must sum to 1"):
            cath_cluster_split(meta, level="H", fractions=(0.5, 0.2, 0.2), seed=42)

    def test_returns_sorted_lists(self):
        meta = self._make_meta()
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        assert train == sorted(train)
        assert val == sorted(val)
        assert test == sorted(test)

    def test_class_stratification_both_classes_represented(self):
        """With enough clusters, both classes should appear in every fold."""
        # Build a bigger synthetic set: 20 H-codes in class 1, 20 in class 3
        meta = {}
        for i in range(20):
            meta[f"c1p{i}"] = {
                "seq": "A", "C": 1, "A": 10,
                "T": f"1.10.{i}", "H": f"1.10.{i}.10",
            }
        for i in range(20):
            meta[f"c3p{i}"] = {
                "seq": "A", "C": 3, "A": 30,
                "T": f"3.30.{i}", "H": f"3.30.{i}.10",
            }
        train, val, test = cath_cluster_split(
            meta, level="H", fractions=(0.6, 0.2, 0.2), seed=42
        )
        for fold, name in [(train, "train"), (val, "val"), (test, "test")]:
            classes = {meta[p]["C"] for p in fold}
            assert classes == {1, 3}, f"{name} missing a class: {classes}"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_seq2oe_splits.py::TestCathClusterSplit -v`
Expected: all FAIL — `cath_cluster_split` not defined

- [ ] **Step 3: Implement cath_cluster_split**

Append to `src/one_embedding/seq2oe_splits.py`:

```python
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
    fold_targets_frac = fractions  # type: ignore[assignment]

    rng = random.Random(seed)

    for cls in sorted(class_to_groups.keys()):
        groups = class_to_groups[cls]
        # Deterministic shuffle of group codes (sorted first, then shuffle)
        group_codes = sorted(groups.keys())
        rng.shuffle(group_codes)

        # Total proteins in this class
        class_total = sum(len(groups[g]) for g in group_codes)
        targets = [f * class_total for f in fold_targets_frac]
        class_counts = [0, 0, 0]

        for g in group_codes:
            members = sorted(groups[g])
            # Which fold is furthest below its target?
            deficits = [t - c for t, c in zip(targets, class_counts)]
            chosen = deficits.index(max(deficits))
            folds[chosen].extend(members)
            class_counts[chosen] += len(members)

    train = sorted(folds[0])
    val = sorted(folds[1])
    test = sorted(folds[2])
    return train, val, test
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_seq2oe_splits.py::TestCathClusterSplit -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/one_embedding/seq2oe_splits.py tests/test_seq2oe_splits.py
git commit -m "feat(exp50): CATH H/T cluster split with per-class greedy stratification"
```

---

### Task 3: Split save/load I/O

**Files:**
- Modify: `src/one_embedding/seq2oe_splits.py`
- Modify: `tests/test_seq2oe_splits.py`

- [ ] **Step 1: Write failing save/load tests**

Append to `tests/test_seq2oe_splits.py`:

```python
class TestSplitIO:
    def test_roundtrip(self, tmp_path):
        train = ["p1", "p2"]
        val = ["p3"]
        test = ["p4", "p5"]
        meta_info = {"level": "H", "seed": 42, "fractions": [0.6, 0.2, 0.2]}

        path = tmp_path / "split.json"
        save_split(path, train, val, test, meta_info)
        assert path.exists()

        loaded_train, loaded_val, loaded_test, loaded_meta = load_split(path)
        assert loaded_train == train
        assert loaded_val == val
        assert loaded_test == test
        assert loaded_meta == meta_info

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "split.json"
        save_split(path, ["p1"], ["p2"], ["p3"], {"level": "T", "seed": 0})
        assert path.exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_seq2oe_splits.py::TestSplitIO -v`
Expected: FAIL — `save_split` not defined

- [ ] **Step 3: Implement save_split / load_split**

Append to `src/one_embedding/seq2oe_splits.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_seq2oe_splits.py::TestSplitIO -v`
Expected: 2 passed

- [ ] **Step 5: Run the whole splits test file**

Run: `uv run pytest tests/test_seq2oe_splits.py -v`
Expected: all passed (14 tests total)

- [ ] **Step 6: Commit**

```bash
git add src/one_embedding/seq2oe_splits.py tests/test_seq2oe_splits.py
git commit -m "feat(exp50): JSON save/load for Seq2OE split files"
```

---

### Task 4: Sanity-check the split on the real CATH20 dataset

**Files:** (none modified — exploratory run)

- [ ] **Step 1: Run a one-off sanity script against the real CATH20 FASTA**

Run:

```bash
uv run python -c "
from src.one_embedding.seq2oe_splits import parse_cath_fasta, cath_cluster_split
meta = parse_cath_fasta('data/external/cath20/cath20_labeled.fasta')
print(f'Parsed {len(meta)} proteins')
for level in ['H', 'T']:
    train, val, test = cath_cluster_split(
        meta, level=level, fractions=(0.8, 0.1, 0.1), seed=42,
    )
    n = len(meta)
    print(f'{level}-split seed=42: '
          f'train={len(train)} ({len(train)/n:.1%}) '
          f'val={len(val)} ({len(val)/n:.1%}) '
          f'test={len(test)} ({len(test)/n:.1%})')
    # Cluster disjointness
    train_clusters = {meta[p][level] for p in train}
    val_clusters = {meta[p][level] for p in val}
    test_clusters = {meta[p][level] for p in test}
    print(f'  {level}-clusters: train={len(train_clusters)} '
          f'val={len(val_clusters)} test={len(test_clusters)} '
          f'overlap_train_test={len(train_clusters & test_clusters)}')
    # Class stratification
    for fold_name, fold in [('train', train), ('val', val), ('test', test)]:
        from collections import Counter
        cls_counts = Counter(meta[p]['C'] for p in fold)
        print(f'  {fold_name} class distribution: {dict(sorted(cls_counts.items()))}')
"
```

Expected output has the form:

```
Parsed 14433 proteins
H-split seed=42: train=~11500 (~80%) val=~1450 (~10%) test=~1450 (~10%)
  H-clusters: train=<N1> val=<N2> test=<N3> overlap_train_test=0
  train class distribution: {1: ..., 2: ..., 3: ..., 4: ..., 6: ...}
  (similar for val/test — all classes present)
T-split seed=42: (similar)
```

Acceptance checks (engineer verifies by eye):
- Total parsed ≈ 14433.
- `overlap_train_test == 0` on cluster sets for both H and T.
- Every fold contains every class that appears in the full dataset.
- Train fraction is within ±5 pp of 80%.

If any check fails, STOP and debug before proceeding. Do not commit anything — this is a read-only sanity check.

- [ ] **Step 2: Decide go/no-go**

If every check passes, proceed to Task 5. If any check fails, stop and debug
the parser or split logic — do not move on.

(This task produces no code changes — it is a sanity gate before the
refactor.)

---

### Task 5: Refactor experiments/50_sequence_to_oe.py — dataset + split dispatch

**Files:**
- Modify: `experiments/50_sequence_to_oe.py`

This is the largest task. Goal: add `--dataset`, `--split`, `--seed`, `--output-root` flags; rewrite `load_data` to dispatch on them; move codec `fit` to run on train embeddings only; rewrite output paths; keep the old random-split behavior available via `--dataset medium5k --split random`.

- [ ] **Step 1: Read the current script to ground the edits**

Run: `uv run cat experiments/50_sequence_to_oe.py | wc -l`
Expected: ~445 lines.

- [ ] **Step 2: Replace argument parser + main entry**

Edit `experiments/50_sequence_to_oe.py`. Replace the `def main()` function (currently starting around line 384) with this version:

```python
def main():
    parser = argparse.ArgumentParser(description="Exp 50: Sequence to One Embedding")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Model stage (1=baseline CNN, 2=deeper)")
    parser.add_argument("--dataset", choices=["cath20", "medium5k"],
                        default="cath20",
                        help="Which embeddings + sequences to use")
    parser.add_argument("--split", choices=["h", "t", "random"],
                        default="h",
                        help="Split strategy: h/t = CATH cluster level, "
                             "random = shuffled 80/10/10")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for split and training RNGs")
    parser.add_argument("--output-root", type=str,
                        default="results/exp50_rigorous",
                        help="Root dir for all outputs")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run with tiny data + 2 epochs")
    parser.add_argument("--eval", action="store_true",
                        help="Run downstream evaluation after training")
    args = parser.parse_args()

    # Validate dataset/split combinations
    if args.dataset == "medium5k" and args.split in ("h", "t"):
        parser.error(
            "--dataset medium5k does not have CATH labels; "
            "use --split random or --dataset cath20"
        )

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: dataset={args.dataset} split={args.split} "
          f"seed={args.seed} stage={args.stage}")

    # Check system load
    load1, load5, _ = os.getloadavg()
    print(f"System load: {load1:.1f} (1m), {load5:.1f} (5m)")
    if load1 > 10:
        print("WARNING: System load >10, consider waiting before training")

    # Load data + build split
    sequences, embeddings, train_ids, val_ids, test_ids, split_meta = load_data(
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )

    # Prepare binary targets — codec centering fit on TRAIN EMBEDDINGS ONLY
    print("\nPreparing binary targets (center + RP896 + sign, train-only fit)...")
    t0 = time.time()
    train_embeddings = {pid: embeddings[pid] for pid in train_ids if pid in embeddings}
    all_targets = prepare_binary_targets_train_fit(
        train_embeddings=train_embeddings,
        all_embeddings=embeddings,
        d_out=896,
        seed=args.seed,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # Sanity: class balance on a train-side protein
    sample_pid = next(iter(all_targets))
    balance = all_targets[sample_pid].mean()
    print(f"  Class balance (sample): {balance:.3f} (ideal=0.5)")

    # Output dir per (split, stage, seed)
    output_root = Path(args.output_root)
    run_dir = output_root / f"{args.split}_split" / f"stage{args.stage}" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist split for reproducibility
    splits_dir = output_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{args.split}_seed{args.seed}.json"
    if not split_path.exists():
        from src.one_embedding.seq2oe_splits import save_split
        save_split(split_path, train_ids, val_ids, test_ids, split_meta)
        print(f"  Saved split to {split_path}")

    # Train
    model, history = train_stage(
        args.stage, sequences, all_targets,
        set(train_ids), set(val_ids), device, args.smoke_test,
        checkpoint_dir=run_dir,
    )

    # Test evaluation
    results = evaluate_bit_accuracy(
        model, sequences, all_targets, set(test_ids), device
    )

    # Per-dimension analysis
    dim_results = analyze_per_dimension(
        model, sequences, all_targets, set(test_ids), device
    )
    results["dim_accuracies"] = dim_results["dim_accuracies"]
    results["config"] = {
        "stage": args.stage,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
    }
    results["best_epoch"] = int(np.argmin(history["val_loss"]) + 1)
    results["best_val_loss"] = float(min(history["val_loss"]))
    results["best_val_bit_acc"] = float(
        history["val_bit_acc"][results["best_epoch"] - 1]
    )

    # Save results
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.eval:
        print("\n[TODO] Downstream evaluation (SS3, disorder, retrieval)")
```

- [ ] **Step 3: Replace load_data with a dispatcher**

Replace the current `load_data` function (currently starting around line 59) with:

```python
def load_data(
    dataset: str,
    split: str,
    seed: int,
    smoke_test: bool = False,
):
    """Load sequences + embeddings and build the requested split.

    Returns:
        (sequences, embeddings, train_ids, val_ids, test_ids, split_meta)
        where `split_meta` is a dict recording the split strategy for JSON I/O.
    """
    if dataset == "cath20":
        fasta_path = DATA / "external" / "cath20" / "cath20_labeled.fasta"
        h5_path = DATA / "residue_embeddings" / "prot_t5_xl_cath20.h5"
    elif dataset == "medium5k":
        fasta_path = DATA / "proteins" / "medium_diverse_5k.fasta"
        h5_path = DATA / "residue_embeddings" / "prot_t5_xl_medium5k.h5"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Loading sequences from {fasta_path}")
    if dataset == "cath20":
        from src.one_embedding.seq2oe_splits import parse_cath_fasta
        cath_meta = parse_cath_fasta(fasta_path)
        sequences = {pid: info["seq"] for pid, info in cath_meta.items()}
    else:
        sequences = read_fasta(fasta_path)
        cath_meta = None

    print(f"Loading ProtT5 embeddings from {h5_path}")
    embeddings = {}
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if smoke_test:
            keys = keys[:50]
        for pid in keys:
            embeddings[pid] = f[pid][:].astype(np.float32)

    # Keep only proteins with both seq and embedding
    common = sorted(set(sequences) & set(embeddings))
    sequences = {k: sequences[k] for k in common}
    embeddings = {k: embeddings[k] for k in common}
    print(f"  {len(common)} proteins with both sequence + embedding")

    # Build the split
    if split == "random":
        rng = np.random.RandomState(seed)
        ids = np.array(common)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_ids = list(ids[:n_train])
        val_ids = list(ids[n_train:n_train + n_val])
        test_ids = list(ids[n_train + n_val:])
        split_meta = {
            "strategy": "random",
            "dataset": dataset,
            "seed": seed,
            "fractions": [0.8, 0.1, 0.1],
        }
    elif split in ("h", "t"):
        from src.one_embedding.seq2oe_splits import cath_cluster_split
        assert cath_meta is not None, "CATH metadata required for h/t split"
        # Restrict cath_meta to proteins that also have embeddings
        filt_meta = {k: cath_meta[k] for k in common if k in cath_meta}
        level = split.upper()
        fractions = (0.8, 0.1, 0.1) if not smoke_test else (0.6, 0.2, 0.2)
        train_ids, val_ids, test_ids = cath_cluster_split(
            filt_meta, level=level, fractions=fractions, seed=seed,
        )
        split_meta = {
            "strategy": f"cath_{level}",
            "dataset": dataset,
            "seed": seed,
            "fractions": list(fractions),
            "level": level,
        }
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"  Split ({split}): "
          f"{len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    return sequences, embeddings, train_ids, val_ids, test_ids, split_meta
```

- [ ] **Step 4: Add prepare_binary_targets_train_fit and train_stage checkpoint arg**

Near the top of the file, after the existing imports, add:

```python
def prepare_binary_targets_train_fit(
    train_embeddings: dict,
    all_embeddings: dict,
    d_out: int = 896,
    seed: int = 42,
) -> dict:
    """Fit OneEmbeddingCodec on train embeddings only, then encode all folds.

    This prevents centering-stat leakage from val/test into the binary targets.
    """
    from src.one_embedding.codec_v2 import OneEmbeddingCodec

    codec = OneEmbeddingCodec(d_out=d_out, quantization="binary", seed=seed)
    codec.fit(train_embeddings)

    targets = {}
    for pid, raw in all_embeddings.items():
        projected = codec._preprocess(raw)
        means = projected.mean(axis=0)
        centered = projected - means[np.newaxis, :]
        bits = (centered > 0).astype(np.uint8)
        targets[pid] = bits

    return targets
```

Then update `train_stage` — currently hard-codes `checkpoint_dir = RESULTS_DIR / f"stage{stage}"` around line 163. Replace that line with:

```python
    if checkpoint_dir is None:
        checkpoint_dir = RESULTS_DIR / f"stage{stage}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

And update the function signature:

```python
def train_stage(
    stage: int,
    sequences: dict,
    targets: dict,
    train_ids: set,
    val_ids: set,
    device: torch.device,
    smoke_test: bool = False,
    checkpoint_dir: Path | None = None,
):
```

- [ ] **Step 5: Update the imports near the top**

The script already imports `argparse`, `json`, `os`, `sys`, `time`, `warnings`, `Path`, `h5py`, `numpy as np`, `torch`, `torch.nn as nn`, `DataLoader`, and from project modules. Verify these imports are all present — no new imports needed at module level (we use lazy `from src.one_embedding.seq2oe_splits import ...` inside functions to avoid import-order surprises on `--dataset medium5k`).

- [ ] **Step 6: Smoke-test the refactored script on medium5k random (back-compat)**

Run:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50_sequence_to_oe.py \
    --dataset medium5k --split random --stage 1 --smoke-test \
    --output-root /tmp/exp50_smoke
```

Expected: completes in ~30 s, prints `Config: dataset=medium5k split=random seed=42 stage=1`, trains 2 epochs on 50 proteins, writes `/tmp/exp50_smoke/random_split/stage1/seed42/results.json` and `/tmp/exp50_smoke/splits/random_seed42.json`, overall_bit_acc ~0.50 (essentially random on 50 proteins × 2 epochs).

- [ ] **Step 7: Smoke-test on cath20 h split**

Run:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50_sequence_to_oe.py \
    --dataset cath20 --split h --stage 1 --smoke-test \
    --output-root /tmp/exp50_smoke
```

Expected: completes in ~30 s, prints `Config: dataset=cath20 split=h seed=42 stage=1`, loads from `cath20_labeled.fasta`, reports split sizes ~30/10/10 (smoke uses 60/20/20 fractions on ~50 proteins), writes `/tmp/exp50_smoke/h_split/stage1/seed42/results.json`.

- [ ] **Step 8: Smoke-test on cath20 t split**

Run:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50_sequence_to_oe.py \
    --dataset cath20 --split t --stage 1 --smoke-test \
    --output-root /tmp/exp50_smoke
```

Expected: similar — no crash, JSON written under `/tmp/exp50_smoke/t_split/stage1/seed42/`.

- [ ] **Step 9: Verify medium5k + h is rejected**

Run:

```bash
uv run python experiments/50_sequence_to_oe.py \
    --dataset medium5k --split h --stage 1 --smoke-test 2>&1 | tail -5
```

Expected: exits non-zero with `error: --dataset medium5k does not have CATH labels`.

- [ ] **Step 10: Clean up smoke output + commit**

```bash
rm -rf /tmp/exp50_smoke
git add experiments/50_sequence_to_oe.py
git commit -m "refactor(exp50): CATH20 dataset + H/T splits + train-only codec fit"
```

---

### Task 6: MMseqs2 leakage audit script

**Files:**
- Create: `experiments/50_leakage_audit.py`

- [ ] **Step 1: Write the audit script**

```python
#!/usr/bin/env python3
"""Exp 50 leakage audit: MMseqs2 search test -> train on the H-split.

Runs once on seed 42 to verify the H-level cluster split produces test
sequences with minimal homology in train. Writes
`results/exp50_rigorous/leakage_audit.json`.

Usage:
    uv run python experiments/50_leakage_audit.py
    uv run python experiments/50_leakage_audit.py --split t --seed 43
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.one_embedding.seq2oe_splits import parse_cath_fasta, cath_cluster_split

DATA = ROOT / "data"
MMSEQS = "/opt/homebrew/bin/mmseqs"


def write_fasta(sequences: dict, path: Path):
    with open(path, "w") as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")


def run_search(query_fa: Path, target_fa: Path, workdir: Path) -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "results.tsv"
    cmd = [
        MMSEQS, "easy-search",
        str(query_fa), str(target_fa), str(out), str(workdir / "tmp"),
        "--min-seq-id", "0.0",
        "--threads", "4",
        "--format-output", "query,target,pident,alnlen,evalue,bits",
        "-v", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"MMseqs2 stderr: {res.stderr}")
        raise RuntimeError("MMseqs2 search failed")
    hits: list[dict] = []
    if out.exists():
        with open(out) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    hits.append({
                        "query": parts[0],
                        "target": parts[1],
                        "pident": float(parts[2]),  # already 0-100
                    })
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["h", "t"], default="h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str,
                        default="results/exp50_rigorous")
    args = parser.parse_args()

    print(f"Loading CATH20 FASTA...")
    cath_fa = DATA / "external" / "cath20" / "cath20_labeled.fasta"
    meta = parse_cath_fasta(cath_fa)
    print(f"  {len(meta)} proteins parsed")

    level = args.split.upper()
    print(f"Building {level}-split (seed={args.seed})...")
    train_ids, val_ids, test_ids = cath_cluster_split(
        meta, level=level, fractions=(0.8, 0.1, 0.1), seed=args.seed,
    )
    print(f"  train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    with tempfile.TemporaryDirectory(prefix="exp50_audit_") as tmp:
        tmp_path = Path(tmp)
        train_fa = tmp_path / "train.fa"
        test_fa = tmp_path / "test.fa"
        write_fasta({k: meta[k]["seq"] for k in train_ids}, train_fa)
        write_fasta({k: meta[k]["seq"] for k in test_ids}, test_fa)

        print(f"Running MMseqs2 easy-search (test -> train)...")
        hits = run_search(test_fa, train_fa, tmp_path / "search")
        print(f"  {len(hits)} total hits")

    # Aggregate max per-query identity
    query_max: dict[str, float] = defaultdict(float)
    for h in hits:
        if h["pident"] > query_max[h["query"]]:
            query_max[h["query"]] = h["pident"]

    n_test = len(test_ids)
    # Every test protein has at least identity-0 "hit" conceptually
    max_per_query = [query_max.get(pid, 0.0) for pid in test_ids]
    arr = np.array(max_per_query)

    thresholds = [20, 25, 30, 40, 50, 60]
    counts = {t: int((arr >= t).sum()) for t in thresholds}
    pcts = {t: float((arr >= t).mean() * 100) for t in thresholds}

    report = {
        "split": level,
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "max_identity_per_test_query": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        },
        "test_queries_with_train_hit_above_pct": counts,
        "test_queries_with_train_hit_above_pct_fraction": pcts,
    }

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"leakage_audit_{level.lower()}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nLeakage audit written to {out_path}")
    print(f"  mean max-identity: {arr.mean():.1f}%")
    print(f"  fraction >= 40% identity: {pcts[40]:.1f}%")

    if pcts[40] > 5.0:
        print(f"WARNING: {pcts[40]:.1f}% of test proteins have a train hit "
              f">= 40% identity — split may be leakier than expected")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the audit (takes a few minutes on 14K × 11K)**

Run:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50_leakage_audit.py --split h --seed 42
```

Expected: prints protein counts, runs MMseqs2, writes
`results/exp50_rigorous/leakage_audit_h_seed42.json`, reports
"fraction >= 40% identity" (expect < 5% since CATH20 is pre-clustered and we
cluster-split further).

- [ ] **Step 3: Inspect the audit report**

```bash
uv run python -c "
import json
r = json.load(open('results/exp50_rigorous/leakage_audit_h_seed42.json'))
import pprint; pprint.pprint(r, sort_dicts=False)
"
```

If `test_queries_with_train_hit_above_pct_fraction['40']` > 5.0, stop and
investigate — something is wrong with the parser or split. Otherwise proceed.

- [ ] **Step 4: Commit**

```bash
git add experiments/50_leakage_audit.py
git commit -m "feat(exp50): MMseqs2 leakage audit for CATH cluster splits"
```

---

### Task 7: Orchestration runner (no-compute shell, launch later)

**Files:**
- Create: `experiments/50b_run_rigorous.py`

- [ ] **Step 1: Write the runner**

```python
#!/usr/bin/env python3
"""Exp 50 rigorous runner: iterate stages × splits × seeds, aggregate.

Invokes `experiments/50_sequence_to_oe.py` for every (stage, split, seed)
triple, one at a time (no MPS parallelism). After all runs complete,
aggregates per-seed results into summary.json per (split, stage) and writes
a final comparison table.

Usage:
    uv run python experiments/50b_run_rigorous.py                # full sweep
    uv run python experiments/50b_run_rigorous.py --stages 1     # only stage 1
    uv run python experiments/50b_run_rigorous.py --splits h     # only H
    uv run python experiments/50b_run_rigorous.py --seeds 42 43  # subset of seeds
    uv run python experiments/50b_run_rigorous.py --aggregate-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "results" / "exp50_rigorous"


def run_one(stage: int, split: str, seed: int) -> bool:
    """Invoke the main experiment script for one (stage, split, seed)."""
    load1, _, _ = os.getloadavg()
    if load1 > 10:
        print(f"[SKIP] System load {load1:.1f} > 10 — aborting this run")
        return False

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv", "run", "python", "experiments/50_sequence_to_oe.py",
        "--dataset", "cath20",
        "--split", split,
        "--stage", str(stage),
        "--seed", str(seed),
        "--output-root", str(OUTPUT_ROOT),
    ]
    print(f"\n{'='*72}")
    print(f"RUN: stage={stage} split={split} seed={seed}")
    print(f"{'='*72}")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"RUN COMPLETE in {elapsed:.0f}s (exit {res.returncode})")
    return res.returncode == 0


def aggregate_seeds(split: str, stage: int) -> dict | None:
    """Aggregate per-seed results.json files into one summary dict."""
    base = OUTPUT_ROOT / f"{split}_split" / f"stage{stage}"
    if not base.exists():
        return None

    per_seed = []
    for seed_dir in sorted(base.glob("seed*")):
        results_file = seed_dir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            per_seed.append(json.load(f))

    if not per_seed:
        return None

    overall = np.array([r["overall_bit_acc"] for r in per_seed])
    per_prot_means = np.array([r["per_protein_mean"] for r in per_seed])
    dim_arrs = np.array([r["dim_accuracies"] for r in per_seed])  # (S, 896)

    # Intersect@60: every seed had that dim > 0.60
    intersect_60 = int((dim_arrs > 0.60).all(axis=0).sum())
    intersect_55 = int((dim_arrs > 0.55).all(axis=0).sum())
    mean_dims = dim_arrs.mean(axis=0)
    mean_60 = int((mean_dims > 0.60).sum())
    mean_55 = int((mean_dims > 0.55).sum())

    summary = {
        "split": split,
        "stage": stage,
        "n_seeds": len(per_seed),
        "seeds": [r["config"]["seed"] for r in per_seed],
        "overall_bit_acc": {
            "mean": float(overall.mean()),
            "std": float(overall.std(ddof=1)) if len(overall) > 1 else 0.0,
            "values": overall.tolist(),
        },
        "per_protein_mean": {
            "mean": float(per_prot_means.mean()),
            "std": float(per_prot_means.std(ddof=1)) if len(per_prot_means) > 1 else 0.0,
        },
        "dims_above_60_intersect": intersect_60,
        "dims_above_55_intersect": intersect_55,
        "dims_above_60_mean": mean_60,
        "dims_above_55_mean": mean_55,
        "best_epochs": [r.get("best_epoch") for r in per_seed],
    }
    summary_path = base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")
    return summary


def write_final_comparison(summaries: list[dict]):
    """Write final_comparison.json and .md."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "final_comparison.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    lines = [
        "# Exp 50 rigorous comparison",
        "",
        "| Stage | Split | Bit acc (mean ± std) | Per-protein mean | dims > 60% (intersect) | dims > 60% (mean) | Seeds |",
        "|:-----:|:-----:|:--------------------:|:----------------:|:----------------------:|:-----------------:|:-----:|",
    ]
    for s in sorted(summaries, key=lambda x: (x["stage"], x["split"])):
        mean = s["overall_bit_acc"]["mean"] * 100
        std = s["overall_bit_acc"]["std"] * 100
        pp_mean = s["per_protein_mean"]["mean"] * 100
        pp_std = s["per_protein_mean"]["std"] * 100
        lines.append(
            f"| {s['stage']} | {s['split']} "
            f"| {mean:.2f} ± {std:.2f} % "
            f"| {pp_mean:.2f} ± {pp_std:.2f} % "
            f"| {s['dims_above_60_intersect']} / 896 "
            f"| {s['dims_above_60_mean']} / 896 "
            f"| {s['n_seeds']} |"
        )
    md_path = OUTPUT_ROOT / "final_comparison.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nFinal comparison written to:\n  {json_path}\n  {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--splits", nargs="+", default=["h", "t"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training, just re-aggregate existing results")
    args = parser.parse_args()

    if not args.aggregate_only:
        for split in args.splits:
            for seed in args.seeds:
                for stage in args.stages:
                    ok = run_one(stage, split, seed)
                    if not ok:
                        print("Run failed — continuing with next configuration")

    # Aggregate
    summaries = []
    for split in args.splits:
        for stage in args.stages:
            s = aggregate_seeds(split, stage)
            if s is not None:
                summaries.append(s)

    if summaries:
        write_final_comparison(summaries)
    else:
        print("No summaries to write — nothing completed successfully.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run aggregate (no training)**

Run:

```bash
uv run python experiments/50b_run_rigorous.py --aggregate-only
```

Expected: prints "No summaries to write" (nothing run yet) and exits cleanly. Just validates the script imports and flags.

- [ ] **Step 3: Commit**

```bash
git add experiments/50b_run_rigorous.py
git commit -m "feat(exp50): runner for rigorous sweep + per-seed aggregation"
```

---

### Task 8: Stage 1 H-split pilot run (one seed, verify before full sweep)

**Files:** (none modified — this is a pilot training run)

- [ ] **Step 1: Verify system load is low enough**

Run: `uptime`
Expected: `load averages: <10 ...` — if load1 > 8, wait for it to drop.

- [ ] **Step 2: Run one Stage 1 H-split training to validate end-to-end**

Run:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50_sequence_to_oe.py \
    --dataset cath20 --split h --stage 1 --seed 42 \
    > results/exp50_rigorous/pilot_stage1_h_seed42.log 2>&1
```

(No `&` — this foregrounds the run so we see it end to end. Expected ~20–30 min on ~11,500 train proteins.)

- [ ] **Step 3: Inspect the pilot result**

Run:

```bash
uv run python -c "
import json
r = json.load(open('results/exp50_rigorous/h_split/stage1/seed42/results.json'))
print(f\"Overall bit acc: {r['overall_bit_acc']:.4f}\")
print(f\"Per-protein: {r['per_protein_mean']:.4f} ± {r['per_protein_std']:.4f}\")
print(f\"Best epoch: {r.get('best_epoch')}\")
import numpy as np
dims = np.array(r['dim_accuracies'])
print(f\"Dims > 60%: {(dims > 0.60).sum()} / 896\")
print(f\"Dims > 55%: {(dims > 0.55).sum()} / 896\")
"
```

Acceptance: no crashes, JSON has all keys, overall_bit_acc in a plausible range (> 0.50). If below 0.52 even on Stage 1 — something is wrong (dead learning, wrong target). Stop and debug before the full sweep.

- [ ] **Step 4: Commit the pilot log (not the checkpoint)**

The per-run checkpoint files (best_model.pt, history.json) are in `results/` which is already gitignored. Only the log file is tracked. Actually — `results/` is gitignored entirely (see CLAUDE.md: "gitignore results/"). So no commit needed. Double check with:

```bash
git status results/exp50_rigorous/
```

Expected: no output (all ignored). If it shows files, check `.gitignore` for `results/`.

No commit for this task.

---

### Task 9: Full sweep (background, overnight)

**Files:** (none modified)

- [ ] **Step 1: Confirm disk space**

Run: `df -h .`
Expected: > 10 GB free on the volume holding this repo. Each run produces a checkpoint file (~20 MB for Stage 2, ~5 MB for Stage 1). 12 runs → < 500 MB total. Not a concern but worth confirming.

- [ ] **Step 2: Launch the full sweep in the background**

We already have seed 42 Stage 1 H-split from Task 8. Skip it and run the remaining 11 configurations. The runner supports arbitrary subsets, but it's simpler to just re-run seed 42 Stage 1 H-split and let the output be overwritten — it takes ~25 min and guarantees a uniform environment for the whole sweep.

Run:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/50b_run_rigorous.py \
    > results/exp50_rigorous/sweep.log 2>&1 &
echo "Launched PID $!"
```

Expected: a PID printed, the process detaches, and `results/exp50_rigorous/sweep.log` starts filling.

- [ ] **Step 3: Poll progress at a reasonable interval**

Over the next several hours, check in with:

```bash
tail -40 results/exp50_rigorous/sweep.log
ls results/exp50_rigorous/h_split/stage*/seed*/results.json 2>/dev/null | wc -l
ls results/exp50_rigorous/t_split/stage*/seed*/results.json 2>/dev/null | wc -l
ps -ef | grep 50_sequence_to_oe | grep -v grep
```

Expected final state: 12 `results.json` files (6 per split × 2 stages × 3 seeds). Do not poll more than once every 15 minutes — use run_in_background for the launch itself.

- [ ] **Step 4: Aggregate and write the final table**

Once all 12 runs are done:

```bash
uv run python experiments/50b_run_rigorous.py --aggregate-only
cat results/exp50_rigorous/final_comparison.md
```

Expected: a 4-row markdown table (2 stages × 2 splits) with mean ± std numbers.

- [ ] **Step 5: Save the headline numbers to memory**

Manually update `~/.claude/projects/-Users-jcoludar-CascadeProjects-ProteEmbedExplorations/memory/project_exp50_seq2oe.md` (existing file) with the rigorous-split numbers, and add a line to `MEMORY.md` linking them. Numbers to capture: Stage 1 H-split mean bit acc, Stage 1 T-split, Stage 2 H-split, Stage 2 T-split, leakage audit summary.

- [ ] **Step 6: Commit nothing from results/ (gitignored), but update the design spec status**

Edit `docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md` — change `**Status:** Approved — proceeding to implementation plan` to `**Status:** Done — see results/exp50_rigorous/final_comparison.md`.

```bash
git add docs/superpowers/specs/2026-04-06-exp50-rigorous-design.md
git commit -m "docs(exp50): mark rigorous CATH-split re-run spec as done"
```

---

## Self-review notes (checked against spec)

- ✅ Dataset loader uses `prot_t5_xl_cath20.h5` + `cath20_labeled.fasta` (spec §Dataset)
- ✅ Binary OE targets via `OneEmbeddingCodec(d_out=896, quantization='binary', abtt_k=0)`, **fit on train only** (Task 5 step 4, spec §Dataset)
- ✅ H-split and T-split both implemented, 80/10/10 of clusters, per-Class greedy stratification (Task 2, spec §Splits)
- ✅ 3 seeds [42, 43, 44] (Task 7, spec §Common)
- ✅ Split JSON files saved under `results/exp50_rigorous/splits/` (Task 5 step 2, spec §Common)
- ✅ MMseqs2 leakage audit on H-split seed 42 with threshold reporting at 20/25/30/40/50/60% (Task 6, spec §Leakage audit)
- ✅ Models and training config unchanged (Task 5 imports `Seq2OE_CNN`, spec §Models §Training)
- ✅ Per-run metrics cover overall_bit_acc, per_protein stats, dim_accuracies, best_epoch/val_loss (Task 5 step 2, spec §Metrics)
- ✅ Aggregation produces `Intersect@60`/`Mean@60` and matching `@55` (Task 7 `aggregate_seeds`, spec §Aggregation)
- ✅ Output directory structure matches spec exactly: `results/exp50_rigorous/{split}_split/stage{N}/seed{seed}/` with `results.json`, `best_model.pt`, `history.json`; `leakage_audit*.json` at root; `splits/` subdir (Task 5–7, spec §Outputs)
- ✅ `medium5k` + `h/t` rejected (Task 5 step 2, spec §Code changes §Refactor — "not supported, error out")
- ✅ Final comparison table in markdown (Task 7 `write_final_comparison`, spec §Success criteria)
- ✅ "Random on cath20" back-compat baseline is valid via `--dataset cath20 --split random` (Task 5, spec §Code changes valid matrix)
- ✅ `PYTHONUNBUFFERED=1` used everywhere (Tasks 5, 6, 7, 8, 9)

No unresolved spec items. No placeholder tasks.
