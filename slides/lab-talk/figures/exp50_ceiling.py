"""F.4 — Exp 50 Stage 1 training curves with Stage 2/3 ceiling annotation.

Stage 1 has a committed history.json with per-epoch train/val bit accuracy.
Stages 2 and 3 do not have committed history JSONs (only best_model.pt
weights), so the multi-stage convergence to ~69% bit accuracy is annotated
from training-log evidence rather than re-plotted.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from _style import CODEC_COLORS, FS_ANNOT, apply_defaults


REPO = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO / "results/exp50/stage1/history.json"
OUT = Path(__file__).parent / "exp50_ceiling.png"


def main():
    apply_defaults()
    h = json.loads(SOURCE.read_text())
    epochs = h["epoch"]
    train_acc = [a * 100 for a in h["train_bit_acc"]]
    val_acc = [a * 100 for a in h["val_bit_acc"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(epochs, train_acc, color="#7E7E7E", lw=1.0, alpha=0.7,
            label=f"Stage 1 train (final {train_acc[-1]:.1f} %)")
    ax.plot(epochs, val_acc, color=CODEC_COLORS["binary"], lw=2.0,
            label=f"Stage 1 val (final {val_acc[-1]:.1f} %)")

    # Random baseline
    ax.axhline(50, ls=":", color="gray", alpha=0.7, lw=1.0)
    ax.text(epochs[-1] * 0.98, 50.7, "random baseline (50 %)",
            color="gray", fontsize=FS_ANNOT, ha="right")

    # Ceiling annotation (from running logs of Stages 2 and 3, not committed JSON)
    ax.axhline(69, ls="--", color="#922B21", alpha=0.7, lw=1.2)
    ax.text(epochs[-1] * 0.98, 69.5,
            "Stage 2 + 3 ceiling ≈ 69 %  (CNN capacity bound)",
            color="#922B21", fontsize=FS_ANNOT, ha="right")

    # Test-set readout from stage1_results.json
    ax.scatter([epochs[-1] + 1.5], [65.42], color=CODEC_COLORS["binary"],
               marker="*", s=180, edgecolor="black", linewidth=0.8, zorder=5,
               label="Stage 1 test bit-acc 65.4 %")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bit accuracy (%)")
    ax.set_title("Exp 50 — sequence → binary OE (CNN, Stage 1)")
    ax.set_ylim(48, 75)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
