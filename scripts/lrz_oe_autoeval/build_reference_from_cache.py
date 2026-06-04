#!/usr/bin/env python
"""Build reference_2000.fasta for OE centering from the cached PBC TRAIN splits.

Leakage-clean: only SET=train sequences are pooled (asserted disjoint from every SET=test id).
Bare-python3 (stdlib only) so it runs on the LRZ login node — the PBC FASTAs are already
downloaded under ~/.cache/biotrainer/autoeval/PBC/supervised/<task>/preprocessed_0_2000/.

  python build_reference_from_cache.py \
      --cache ~/.cache/biotrainer/autoeval/PBC/supervised \
      --out  .../oe_autoeval_lrz/reference_2000.fasta  --n 2000
"""
import argparse
import random
from pathlib import Path

TASKS = ["conservation", "disorder", "scl", "secondary_structure"]


def read_fasta_with_set(path):
    sid = split = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(buf), split
                parts = line[1:].split()
                sid, split, buf = parts[0], "train", []
                for p in parts[1:]:
                    if p.upper().startswith("SET="):
                        split = p.split("=", 1)[1].lower()
            else:
                buf.append(line.strip())
    if sid is not None:
        yield sid, "".join(buf), split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help=".../PBC/supervised")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    train, test = {}, set()
    for t in TASKS:
        fa = Path(args.cache).expanduser() / t / "preprocessed_0_2000" / f"{t}.fasta"
        if not fa.exists():
            print(f"WARN: missing {fa}")
            continue
        for sid, seq, split in read_fasta_with_set(fa):
            if split == "test":
                test.add(sid)
            elif split == "train" and len(seq) >= 4:   # L>=4 guard for protein_vec
                train[sid] = seq

    # A protein can be train in one task but test in another (PBC tasks have independent
    # splits). Exclude any id that is test in ANY task -> reference disjoint from every test set.
    n_dropped = sum(1 for k in train if k in test)
    train = {k: v for k, v in train.items() if k not in test}
    ids = sorted(train)
    rng = random.Random(42)
    if len(ids) > args.n:
        ids = sorted(rng.sample(ids, args.n))
    assert not (set(ids) & test), "reference still leaks into test"

    with open(args.out, "w") as o:
        for sid in ids:
            o.write(f">{sid}\n{train[sid]}\n")
    print(f"reference: {len(ids)} train seqs (dropped {n_dropped} test-in-other-task; "
          f"{len(train)} clean train, {len(test)} test ids) -> {args.out}")


if __name__ == "__main__":
    main()
