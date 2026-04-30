"""Exp 51 — PolarQuant (binary + per-residue magnitude) eval.

Hypothesis: adding one fp16 magnitude scalar per residue on top of the
binary OE codec recovers some of the disorder retention gap (Exp 47:
binary 94.9 % vs int4 98.2 %, vs PQ M=224 95.4 %) while keeping the
codebook-free, fast-encoding properties of binary.

Tests four configs on ProtT5-XL:
  - binary 896d        (current default — baseline)
  - binary_magnitude   (new — the experimental tier)
  - int4 896d          (upper-reference)
  - PQ M=224 896d      (current quality leader)

Tasks (Exp 47 protocol, BCa CIs, paired bootstrap retention):
  - CheZOD disorder retention (the headline)
  - CB513 SS3 retention
  - SCOPe-5K retrieval Ret@1 (sanity — should be ~lossless across all)

Output:
    results/exp51_polarquant.json
    + per-PLM exp47_sweep_prot_t5_full.json (overwritten if same configs match)

Usage:
    uv run python experiments/51_polarquant_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "43_rigorous_benchmark"))

# Re-use Exp 47's machinery
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("exp47", str(ROOT / "experiments" / "47_codec_sweep.py"))
exp47 = module_from_spec(_spec)
_spec.loader.exec_module(exp47)

CodecConfig = exp47.CodecConfig
run_sweep = exp47.run_sweep
print_sweep_table = exp47.print_sweep_table


CONFIGS = [
    CodecConfig("binary-896",            896, "binary",
                description="baseline: RP 896d, binary (~37x)"),
    CodecConfig("binary_magnitude-896",  896, "binary_magnitude",
                description="EXP 51: RP 896d, binary + per-residue fp16 magnitude (~32x)"),
    CodecConfig("int4-896",              896, "int4",
                description="upper-ref: RP 896d, int4 (~9x)"),
    CodecConfig("pq224-896",             896, "pq", pq_m=224,
                description="quality-leader: RP 896d, PQ M=224 (~18x, codebook required)"),
]


def main() -> None:
    plm_name = "prot_t5_full"
    print(f"Exp 51 — PolarQuant eval on {plm_name}")
    print(f"Configs ({len(CONFIGS)}):")
    for c in CONFIGS:
        print(f"  {c.name:<25} {c.compression_ratio:>5} — {c.description}")

    t0 = time.time()
    results = run_sweep(plm_name, CONFIGS, bootstrap_n=exp47.BOOTSTRAP_N)
    elapsed = time.time() - t0

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp51_polarquant.json"
    with open(out_path, "w") as f:
        json.dump({
            "plm": plm_name,
            "configs": results,
            "wall_time_s": elapsed,
        }, f, indent=2, default=float)

    print_sweep_table(plm_name, results)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
