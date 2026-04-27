"""Print shapes and on-disk sizes of the three demo forms.

The same 10 CASP12 proteins are stored in each of:
  raw/        per-residue,  fp16, (L, 1024)
  mean_pool/  per-protein,  fp32, (1024,)
  ours/       per-residue,  binary, (L, 896) + protein-vec (3584,) fp16

Run from the repo root:
    uv run python demo/inspect.py
"""
from __future__ import annotations

from pathlib import Path

import h5py

DEMO = Path(__file__).resolve().parent
PATHS = [
    ("raw       (per-residue, fp16)     ", DEMO / "raw" / "casp12_first10.h5"),
    ("mean_pool (per-protein, fp32)     ", DEMO / "mean_pool" / "casp12_first10.h5"),
    ("ours      (binary 896d, no codebk)", DEMO / "ours" / "casp12_first10.one.h5"),
]


def main() -> None:
    print(f"{'form':38s} {'on-disk':>10s}   sample shape")
    print("-" * 78)
    for label, path in PATHS:
        if not path.exists():
            print(f"{label:38s} {'MISSING':>10s}   (run demo/build.py first)")
            continue
        size_kb = path.stat().st_size / 1024
        with h5py.File(path) as f:
            keys = list(f.keys())
            sample_pid = keys[0] if "ours" not in str(path) else keys[0]
            obj = f[sample_pid]
            if hasattr(obj, "shape"):
                shape = obj.shape
            else:
                shape = "(group with " + ", ".join(obj.keys()) + ")"
        print(f"{label:38s} {size_kb:8.1f} KB   {sample_pid}: {shape}")


if __name__ == "__main__":
    main()
