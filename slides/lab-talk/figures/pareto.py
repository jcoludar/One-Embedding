"""F.2 — Pareto plot: compression × mean retention from Exp 47 (ProtT5-XL).

Six configs (lossless / fp16 / int4 / PQ M=224 / PQ M=128 / binary) at d_out=896.
Mean retention = mean over (SS3, SS8, Ret@1, Disorder).
CI half-width from the BCa CIs in the source JSON, propagated as quadrature mean.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import CODEC_COLORS, FS_ANNOT, apply_defaults


REPO = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO / "data/benchmarks/rigorous_v1/exp47_sweep_prot_t5_full.json"
OUT = Path(__file__).parent / "pareto.png"


def main():
    apply_defaults()
    data = json.loads(SOURCE.read_text())

    # Six "headline" configs in CLAUDE.md table order
    headline_configs = [
        ("lossless-1024", "lossless 1024d", "lossless", "o", 60),
        ("fp16-896",      "fp16 896d",       "fp16",     "s", 60),
        ("int4-896",      "int4 896d",       "int4",     "^", 60),
        ("pq224-896",     "PQ M=224 896d",   "pq_max",   "*", 200),     # bold marker
        ("pq128-896",     "PQ M=128 896d",   "pq_aggressive", "v", 60),
        ("binary-896",    "binary 896d",     "binary",   "D", 200),    # bold marker
    ]

    by_config = {c["config"]: c for c in data["configs"]}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for cfg_id, label, color_key, marker, size in headline_configs:
        c = by_config.get(cfg_id)
        if c is None:
            print(f"  [WARN] config {cfg_id} not found; skipping")
            continue

        # x = compression factor (parse "37x" → 37)
        x = float(str(c["compression"]).rstrip("x"))

        # y = mean retention across 4 task families
        ret_vals = [c["ss3"]["retention"], c["ss8"]["retention"],
                    c["ret1"]["retention"], c["disorder"]["retention"]]
        y = float(np.mean(ret_vals))

        # CI half-width = mean of per-task half-widths (rough propagation)
        halfwidths = []
        for task_key in ["ss3", "ss8", "ret1", "disorder"]:
            ci = c[task_key].get("ci")
            if ci and len(ci) == 2:
                halfwidths.append((ci[1] - ci[0]) / 2)
        yerr = float(np.mean(halfwidths)) if halfwidths else 0.0

        ax.errorbar(x, y, yerr=yerr, fmt=marker, markersize=np.sqrt(size),
                    color=CODEC_COLORS[color_key], capsize=3, linewidth=1.5,
                    label=label)

        # Annotation slightly offset
        ax.annotate(label, (x, y), xytext=(8, -2), textcoords="offset points",
                    fontsize=FS_ANNOT, color=CODEC_COLORS[color_key])

    ax.set_xscale("log")
    ax.set_xlabel("Compression factor (×, log scale)")
    ax.set_ylabel("Mean retention across 4 task families (%)")
    ax.set_title("Codec sweep — Exp 47, ProtT5-XL on 4-task mean retention")
    ax.axhline(100, ls="--", color="gray", alpha=0.5, lw=0.8)
    ax.text(1.5, 100.3, "raw baseline (100 %)", color="gray", fontsize=FS_ANNOT)
    ax.set_xlim(1.5, 50)
    ax.set_ylim(94, 102)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
