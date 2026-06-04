#!/usr/bin/env python
"""Build the centering reference FASTA from the UNION of PBC TRAIN splits (PLAN §R7).

Exact-id-disjoint from every PBC test split by construction (we only pool train ids). Also
runs the G4 scan (any sequence with L<4 → the per-sequence wrapper would reject it). Run on
the login node after 02_prestage_pbc.py.

  source config.sh
  /work/venv/bin/python 03_build_reference.py --cache "$BIOTRAINER_CACHE" \
      --out /work/reference_2000.fasta --n 2000

CONFIRM on-cluster: the on-disk PBC layout (per-task fasta + split annotation). The reader
below assumes biotrainer's standard FASTA with SET=train/val/test in the header; adjust to the
actual PBC files once visible.
"""
import argparse
import sys
from pathlib import Path

REPO = "/work/ProteEmbedExplorations"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.oe_autoeval.reference_set import sample_reference_ids  # noqa: E402

PBC_TASKS = [
    "conservation", "secondary_structure", "membrane",
    "disorder_chezod", "disorder_trizod",
    "frustration-classification", "frustration-regression",
    "phages", "scl",
]


def read_pbc_fasta(path):
    """Yield (id, seq, split) for a biotrainer-style FASTA (SET=train|val|test in header)."""
    sid = seq = split = None
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(buf), split
                header = line[1:]
                sid = header.split()[0]
                split = "train"
                for tok in header.split():
                    if tok.upper().startswith("SET="):
                        split = tok.split("=", 1)[1].lower()
                buf = []
            else:
                buf.append(line.strip())
    if sid is not None:
        yield sid, "".join(buf), split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    cache = Path(args.cache)
    train, test_ids, by_id = {}, set(), {}
    short = []
    for task in PBC_TASKS:
        # CONFIRM the actual filename pattern under the PBC cache.
        for fasta in cache.rglob(f"*{task}*.fasta"):
            for sid, seq, split in read_pbc_fasta(fasta):
                if len(seq) < 4:
                    short.append((task, sid, len(seq)))
                if split == "test":
                    test_ids.add(sid)
                elif split == "train":
                    train[sid] = seq
                    by_id[sid] = seq

    ref_ids = sample_reference_ids(list(train), n=args.n, seed=42)
    # Hard disjointness assert (PLAN §R2/§R7): no reference id is in any test split.
    overlap = set(ref_ids) & test_ids
    assert not overlap, f"reference leaks {len(overlap)} ids into PBC test splits"

    with open(args.out, "w") as out:
        for sid in ref_ids:
            out.write(f">{sid}\n{by_id[sid]}\n")
    print(f"reference set: {len(ref_ids)} seqs -> {args.out} (disjoint from {len(test_ids)} test ids)")
    if short:
        print(f"G4: {len(short)} sequences with L<4 found across PBC (per-sequence wrapper will reject): {short[:10]}")
    else:
        print("G4: no L<4 sequences in PBC — per-sequence guard will not trip.")


if __name__ == "__main__":
    main()
