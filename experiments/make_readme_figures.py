#!/usr/bin/env python3
"""Generate README figures: architecture, key results, scaling, failure analysis."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "grey": "#6B7280",
    "light": "#E5E7EB",
}


def fig_main_results():
    """Bar chart comparing compressed vs original across key metrics."""
    metrics = ["Retrieval\nP@1", "SS3\n(Q3)", "SS8\n(Q8)", "ToxFam\nF1"]
    compressed_256 = [0.795, 0.821, 0.692, 0.956]
    compressed_128 = [0.777, None, None, None]
    original = [0.830, 0.834, 0.704, 0.941]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_orig = ax.bar(x - width, original, width, label="Original (1024d)",
                       color=COLORS["light"], edgecolor=COLORS["grey"], linewidth=1.2)
    bars_256 = ax.bar(x, compressed_256, width, label="Compressed d256 (4x)",
                      color=COLORS["primary"], edgecolor="white", linewidth=0.5)

    # d128 only for retrieval
    d128_vals = [compressed_128[0]]
    ax.bar([x[0] + width], d128_vals, width, label="Compressed d128 (8x)",
           color=COLORS["secondary"], edgecolor="white", linewidth=0.5)

    # Highlight where compressed beats original
    ax.annotate("compressed\nbeats original", xy=(x[3], 0.956), xytext=(x[3] + 0.35, 0.975),
                fontsize=8, ha="center", color=COLORS["accent"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.5))

    ax.set_ylabel("Score")
    ax.set_title("Compressed vs Original PLM Embeddings (ProtT5-XL)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=1.0, color=COLORS["light"], linestyle="--", linewidth=0.8)

    # Value labels on bars
    for bars in [bars_orig, bars_256]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, color=COLORS["grey"])

    out = PLOTS_DIR / "readme_main_results.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig_scaling_curve():
    """Scaling curve: training data fraction vs Ret@1."""
    results = json.load(open("data/benchmarks/scaling_ablation_results.json"))
    s1 = sorted([r for r in results if r["step"] == "S1"], key=lambda r: r["fraction"])

    fracs = [r["fraction"] * 100 for r in s1]
    rets = [r["retrieval"]["precision@1"] for r in s1]
    n_proteins = [r["n_train_proteins"] for r in s1]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(fracs, rets, "o-", color=COLORS["primary"], linewidth=2.5, markersize=8,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor=COLORS["primary"])

    # Annotate saturation point
    ax.axvline(x=75, color=COLORS["warning"], linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate("saturates at 75%\n(~1200 proteins)", xy=(75, 0.798),
                xytext=(45, 0.76), fontsize=9, color=COLORS["warning"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["warning"], lw=1.5))

    # Protein counts
    for f, r, n in zip(fracs, rets, n_proteins):
        ax.text(f, r + 0.008, f"n={n}", ha="center", fontsize=7.5, color=COLORS["grey"])

    ax.set_xlabel("Training Data Fraction (%)")
    ax.set_ylabel("Family Retrieval P@1")
    ax.set_title("Data Efficiency: How Much Contrastive Data Is Needed?")
    ax.set_xlim(0, 108)
    ax.set_ylim(0.6, 0.83)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = PLOTS_DIR / "readme_scaling.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig_ablations():
    """Horizontal bar chart of ablation impact on Ret@1."""
    labels = ["No Residual\nConnection", "No LayerNorm", "No Decoder\nFreeze"]
    deltas = [-0.169, -0.015, -0.001]
    colors = [COLORS["danger"], COLORS["warning"], COLORS["accent"]]

    fig, ax = plt.subplots(figsize=(7, 3))

    y = np.arange(len(labels))
    bars = ax.barh(y, deltas, height=0.5, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in Ret@1 vs Baseline (0.808)")
    ax.set_title("Architecture Ablations: What Matters?")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlim(-0.20, 0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, d in zip(bars, deltas):
        ax.text(bar.get_width() - 0.005, bar.get_y() + bar.get_height()/2,
                f"{d:+.3f}", ha="right", va="center", fontsize=10, fontweight="bold",
                color="white" if abs(d) > 0.05 else COLORS["grey"])

    # Annotation for critical finding
    ax.annotate("CRITICAL", xy=(-0.169, 0), xytext=(-0.12, 0.8),
                fontsize=9, color=COLORS["danger"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["danger"], lw=1.5))

    out = PLOTS_DIR / "readme_ablations.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig_failure_by_class():
    """Per-class retrieval performance bar chart."""
    classes = ["f\nmembrane", "b\nbeta", "d\nalpha+beta", "c\nalpha/beta",
               "a\nalpha", "g\nsmall", "e\nmulti-domain"]
    means = [0.936, 0.839, 0.835, 0.818, 0.794, 0.762, 0.685]
    n_fam = [7, 42, 36, 49, 53, 14, 9]

    # Color gradient from good to bad
    norm_vals = [(v - 0.65) / (0.95 - 0.65) for v in means]
    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(classes))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5, width=0.65)

    # Family count labels
    for i, (bar, n) in enumerate(zip(bars, n_fam)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{means[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, 0.62,
                f"({n} fam)", ha="center", fontsize=7.5, color=COLORS["grey"])

    ax.set_ylabel("Mean Family Retrieval P@1")
    ax.set_title("Retrieval Performance by SCOPe Structural Class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0.58, 1.0)
    ax.axhline(y=0.795, color=COLORS["grey"], linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(classes) - 0.5, 0.798, "overall mean", fontsize=8,
            color=COLORS["grey"], ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = PLOTS_DIR / "readme_failure_by_class.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig_architecture():
    """Architecture diagram for the ChannelCompressor pipeline."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", facecolor=COLORS["primary"],
                     edgecolor="white", alpha=0.9)
    decoder_style = dict(boxstyle="round,pad=0.4", facecolor=COLORS["secondary"],
                         edgecolor="white", alpha=0.9)
    latent_style = dict(boxstyle="round,pad=0.5", facecolor=COLORS["accent"],
                        edgecolor="white", alpha=0.9)
    input_style = dict(boxstyle="round,pad=0.4", facecolor=COLORS["light"],
                       edgecolor=COLORS["grey"])

    text_kw = dict(ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    dim_kw = dict(ha="center", va="center", fontsize=8, color=COLORS["grey"])

    # Input
    ax.text(2, 5.3, "PLM Embeddings\n(B, L, 1024)", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLORS["grey"], bbox=input_style)

    # Encoder
    ax.text(2, 4.2, "LayerNorm + Linear", **text_kw, bbox=box_style)
    ax.text(3.8, 4.2, "1024 -> 512", **dim_kw)
    ax.annotate("", xy=(2, 4.55), xytext=(2, 5.0),
                arrowprops=dict(arrowstyle="->", color=COLORS["grey"], lw=1.5))

    ax.text(2, 3.3, "LayerNorm + GELU + Dropout", **text_kw, bbox=box_style)
    ax.annotate("", xy=(2, 3.65), xytext=(2, 3.95),
                arrowprops=dict(arrowstyle="->", color=COLORS["grey"], lw=1.5))

    ax.text(2, 2.4, "Linear + Residual", **text_kw, bbox=box_style)
    ax.text(3.8, 2.4, "512 -> 256", **dim_kw)
    ax.annotate("", xy=(2, 2.75), xytext=(2, 3.05),
                arrowprops=dict(arrowstyle="->", color=COLORS["grey"], lw=1.5))

    # Latent
    ax.text(5, 2.4, "Latent\n(B, L, 256)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", bbox=latent_style)
    ax.annotate("", xy=(4.1, 2.4), xytext=(3.1, 2.4),
                arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=2))

    # Decoder
    ax.text(8, 4.2, "Linear", **text_kw, bbox=decoder_style)
    ax.text(9.3, 4.2, "256 -> 512", **dim_kw)

    ax.text(8, 3.3, "LayerNorm + GELU + Dropout", **text_kw, bbox=decoder_style)

    ax.text(8, 2.4, "Linear", **text_kw, bbox=decoder_style)
    ax.text(9.3, 2.4, "512 -> 1024", **dim_kw)

    ax.annotate("", xy=(8, 3.65), xytext=(8, 3.95),
                arrowprops=dict(arrowstyle="->", color=COLORS["grey"], lw=1.5))
    ax.annotate("", xy=(8, 2.75), xytext=(8, 3.05),
                arrowprops=dict(arrowstyle="->", color=COLORS["grey"], lw=1.5))

    # Latent -> Decoder
    ax.annotate("", xy=(6.9, 3.0), xytext=(5.8, 2.7),
                arrowprops=dict(arrowstyle="->", color=COLORS["secondary"], lw=2))

    # Output
    ax.text(8, 1.5, "Reconstructed\n(B, L, 1024)", ha="center", va="center",
            fontsize=11, fontweight="bold", color=COLORS["grey"], bbox=input_style)
    ax.annotate("", xy=(8, 1.85), xytext=(8, 2.15),
                arrowprops=dict(arrowstyle="->", color=COLORS["grey"], lw=1.5))

    # Labels
    ax.text(2, 5.8, "ENCODER", ha="center", fontsize=12, fontweight="bold",
            color=COLORS["primary"])
    ax.text(8, 5.8, "DECODER (mirror)", ha="center", fontsize=12, fontweight="bold",
            color=COLORS["secondary"])
    ax.text(5, 1.5, "4x compressed\nper-residue", ha="center", fontsize=9,
            color=COLORS["accent"], fontstyle="italic")

    # Training pipeline at bottom
    ax.text(5, 0.5, "Stage 1: Unsupervised reconstruction (200 epochs)  ->  "
            "Stage 2: Contrastive fine-tuning with InfoNCE (100 epochs)",
            ha="center", va="center", fontsize=9, color=COLORS["grey"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9FAFB",
                      edgecolor=COLORS["light"]))

    out = PLOTS_DIR / "readme_architecture.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating README figures...\n")
    fig_main_results()
    fig_scaling_curve()
    fig_ablations()
    fig_failure_by_class()
    fig_architecture()
    print("\nDone! All figures saved to data/plots/")
