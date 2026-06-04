#!/usr/bin/env python
"""Run ONE autoeval arm (PLM × mode) on LRZ: embed reference → fit codec → autoeval PBC.

Invoked inside the container by arm.sbatch. Uses src.oe_autoeval.driver (unit-tested wiring)
+ the real biotrainer autoeval_pipeline. The per-seed probe training / per-item-prediction
harvest is a SEPARATE step (05_analyze.py via the biotrainer-direct path, PLAN §R3) because
autoeval 1.4.0 cannot vary the probe seed; this script produces the embeddings + the default
single-seed report per arm.

  python run_arm.py --embedder facebook/esm2_t6_8M_UR50D --mode oe --native-d 320 \
      --precision fp32 --label ESM2-8M-ONE --reference /work/reference_2000.fasta --out /work/out/ESM2-8M-ONE
"""
import argparse
import sys
from pathlib import Path

# Repo is pip install -e'd in the venv, but keep the path insert for robustness.
REPO = "/work/ProteEmbedExplorations"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from src.oe_autoeval import driver  # noqa: E402


def read_fasta(path):
    seqs = []
    sid, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if sid is not None:
                    seqs.append("".join(buf))
                sid, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if sid is not None:
        seqs.append("".join(buf))
    return seqs


def embed_reference(svc, sequences):
    """Embed the reference set (raw) → {seq: (L, D)} for codec.fit (keys are arbitrary)."""
    embs = {}
    for rec, t in svc.generate_embeddings(input_data=sequences, reduce=False):
        embs[rec.seq] = np.asarray(t.cpu().numpy(), dtype=np.float32)
    return embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True, help="HF id, e.g. facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--mode", required=True, choices=["raw", "oe", "oe_norp"])
    ap.add_argument("--native-d", type=int, required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16"])
    ap.add_argument("--label", required=True, help="display/cache label, e.g. ESM2-8M-ONE")
    ap.add_argument("--reference", required=True, help="centering reference FASTA")
    ap.add_argument("--out", required=True, help="output_dir for autoeval")
    args = ap.parse_args()

    from biotrainer.autoeval import autoeval_pipeline

    svc = driver.build_service(args.embedder, args.precision)
    ref_embs = None
    if args.mode != "raw":
        ref_embs = embed_reference(svc, read_fasta(args.reference))
        print(f"[{args.label}] fitted centering on {len(ref_embs)} reference proteins")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    driver.run(
        embed_service=svc,
        autoeval_pipeline=autoeval_pipeline,
        mode=args.mode,
        native_d=args.native_d,
        precision=args.precision,
        label=args.label,
        output_dir=args.out,
        reference_embeddings=ref_embs,
    )
    print(f"[{args.label}] autoeval complete -> {args.out}")


if __name__ == "__main__":
    main()
