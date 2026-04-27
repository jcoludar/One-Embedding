"""Build a small 3-form demo from CASP12 ProtT5 embeddings.

Forms (same 10 proteins in each):

  raw/        per-residue (L, 1024) fp16 — what the PLM emits per amino acid
  mean_pool/  per-protein (1024,)  fp32 — naive baseline (channel-wise mean over L)
  ours/       per-residue (L, 896) binary + protein vec (3584,) fp16
              — our codec, ~37x compression, no codebook

Run from the repo root:
    uv run python demo/build.py

Then:
    uv run python demo/show.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.one_embedding.codec_v2 import OneEmbeddingCodec  # noqa: E402

RAW_SRC = REPO / "data/residue_embeddings/prot_t5_xl_casp12.h5"
DEMO = REPO / "demo"
N = 10


def main() -> None:
    if not RAW_SRC.exists():
        sys.exit(f"Missing source embeddings: {RAW_SRC}\n"
                 f"Run experiments/01_extract_residue_embeddings.py first.")

    with h5py.File(RAW_SRC) as f:
        pids = list(f.keys())[:N]
        raw = {pid: f[pid][:] for pid in pids}
    print(f"Read {len(raw)} proteins from {RAW_SRC.name}")

    out_raw = DEMO / "raw" / "casp12_first10.h5"
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_raw, "w") as f:
        for pid, m in raw.items():
            f.create_dataset(pid, data=m.astype(np.float16), compression="gzip")
    print(f"  raw/        {out_raw.relative_to(REPO)}")

    out_mp = DEMO / "mean_pool" / "casp12_first10.h5"
    out_mp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_mp, "w") as f:
        for pid, m in raw.items():
            f.create_dataset(pid, data=m.mean(axis=0).astype(np.float32))
    print(f"  mean_pool/  {out_mp.relative_to(REPO)}")

    codec = OneEmbeddingCodec()
    codec.fit(raw)
    out_ours = DEMO / "ours" / "casp12_first10.one.h5"
    out_ours.parent.mkdir(parents=True, exist_ok=True)
    codec.encode_h5_to_h5(str(out_raw), str(out_ours))
    print(f"  ours/       {out_ours.relative_to(REPO)}")


if __name__ == "__main__":
    main()
