"""F.3 — 5-PLM × 4-task retention heatmap from Exp 46.

Codec: center + RP896 + PQ M=224 (~18×).
Cells = retention %, diverging colormap centered at 100 %.
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from _style import FS_ANNOT, RETENTION_CMAP, apply_defaults


REPO = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO / "data/benchmarks/rigorous_v1/exp46_multi_plm_results.json"
OUT = Path(__file__).parent / "multi_plm_heatmap.png"


def main():
    apply_defaults()
    data = json.loads(SOURCE.read_text())

    # Order PLMs as in the talk-table (best disorder ret first)
    plm_order = [
        ("prostt5",      "ProstT5"),
        ("prot_t5_full", "ProtT5-XL"),
        ("esmc_600m",    "ESM-C 600M"),
        ("ankh_large",   "ANKH-large"),
        ("esm2_650m",    "ESM2-650M"),
    ]
    task_order = [("ss3", "SS3"), ("ss8", "SS8"), ("ret1", "Ret@1"), ("disorder", "Disorder")]

    matrix = np.zeros((len(plm_order), len(task_order)))
    for i, (plm_key, _) in enumerate(plm_order):
        for j, (task_key, _) in enumerate(task_order):
            matrix[i, j] = data[plm_key][task_key]["retention"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Center the diverging colormap at 100 %
    vmin, vmax = 90.0, 105.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=100.0, vmax=vmax)
    im = ax.imshow(matrix, cmap=RETENTION_CMAP, norm=norm, aspect="auto")

    ax.set_xticks(range(len(task_order)), [t[1] for t in task_order])
    ax.set_yticks(range(len(plm_order)), [p[1] for p in plm_order])

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            # Choose text color for contrast
            tc = "white" if abs(val - 100) > 4 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=11, color=tc, weight="medium")

    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, label="Retention (%)", shrink=0.85)
    cbar.ax.tick_params(labelsize=FS_ANNOT)

    ax.set_title("Multi-PLM validation — Exp 46 (PQ M=224 896d, ~18× compression)")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
