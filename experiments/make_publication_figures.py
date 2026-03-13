#!/usr/bin/env python3
"""Generate publication-quality figures for the universal codec paper.

6 figures covering:
  1. Codec retrieval benchmark (main result)
  2. Per-residue task retention
  3. Retrieval vs per-residue trade-off scatter
  4. Cross-PLM comparison
  5. Biology & hierarchy multi-panel
  6. Trained ChannelCompressor addendum
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Output & style
# ---------------------------------------------------------------------------
FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Consistent codec color palette
CODEC_COLORS = {
    # Pooling-only (amber family)
    "mean_pool":         "#D97706",
    "max_pool":          "#F59E0B",
    "mean_max":          "#FBBF24",
    "mean_max_euc":      "#FDE68A",
    # D-compression (green family)
    "rp512":             "#059669",
    "fh512":             "#10B981",
    # Smart pooling (blue family)
    "dct_K4":            "#2563EB",
    "cosine_deviation":  "#60A5FA",
    # Chained (purple family)
    "rp512_dct_K4":      "#7C3AED",
    "fh512_dct_K4":      "#A78BFA",
    "rp512_mean_max":    "#C4B5FD",
    "rp512_mean_max_euc":"#8B5CF6",
    # Trained (gold)
    "trained_cc":        "#B45309",
    # Reference lines
    "ground_zero":       "#6B7280",
}

CATEGORY_COLORS = {
    "Pooling-only":   "#D97706",
    "D-compression":  "#059669",
    "Smart pooling":  "#2563EB",
    "Chained":        "#7C3AED",
    "Trained":        "#B45309",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normal_ci(p, n, z=1.96):
    """95% CI half-width for a proportion p with n observations."""
    return z * np.sqrt(p * (1 - p) / n)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load all benchmark JSON files and return a data dict."""
    data = {}

    # Exp 25: universal codec
    uc = load_json("data/benchmarks/universal_codec_results.json")
    data["uc_retrieval"] = uc["results"]["retrieval"]
    data["uc_hierarchy"] = uc["results"]["hierarchy"]
    data["uc_biology"] = uc["results"]["biology"]
    data["uc_per_residue"] = uc["results"]["per_residue"]

    # Exp 26: chained codecs
    cc = load_json("data/benchmarks/chained_codec_results.json")
    data["chained_rows"] = cc["results"]["summary_rows"]
    data["chained_retrieval"] = cc["results"].get("retrieval", {})

    # PLM benchmark suite (raw baselines)
    plm = load_json("data/benchmarks/plm_benchmark_suite_results.json")
    data["plm_retrieval"] = plm["results"]["retrieval"]
    data["plm_hierarchy"] = plm["results"]["hierarchy"]
    data["plm_per_residue"] = plm["results"]["per_residue"]
    data["plm_biology"] = plm["results"]["biology"]

    # Robust validation (trained CC, seeds 123/456)
    rv = load_json("data/benchmarks/robust_validation_results.json")
    data["robust_val"] = rv

    # Channel compression (trained CC, seed 42 from Exp 11)
    ch = load_json("data/benchmarks/channel_compression_results.json")
    data["channel_comp"] = ch

    return data


# ---------------------------------------------------------------------------
# Figure 1: Codec Retrieval Benchmark
# ---------------------------------------------------------------------------

def fig_codec_retrieval(data):
    """Bar chart: ~12 codecs sorted by ProtT5 Ret@1 with 95% CI error bars."""
    n = 850  # n_queries

    # Collect ProtT5 retrieval results
    codecs = [
        ("mean_pool",         data["uc_retrieval"]["prot_t5_xl_mean_pool"]["family_ret1"]),
        ("max_pool",          data["uc_retrieval"]["prot_t5_xl_max_pool"]["family_ret1"]),
        ("[mean|max]",        data["uc_retrieval"]["prot_t5_xl_mean_max"]["family_ret1"]),
        ("[mean|max] euc",    data["uc_retrieval"]["prot_t5_xl_mean_max_euclidean"]["family_ret1"]),
        ("dct K=4",           data["uc_retrieval"]["prot_t5_xl_dct_K4"]["family_ret1"]),
        ("rp512",             data["uc_retrieval"]["prot_t5_xl_rp512"]["family_ret1"]),
        ("fh512",             data["uc_retrieval"]["prot_t5_xl_fh512"]["family_ret1"]),
        ("cosine dev",        data["chained_retrieval"]["prot_t5_xl_cosine_deviation"]["family_ret1"]),
        ("rp512+dct K4",     data["chained_retrieval"]["prot_t5_xl_rp512_dct_K4"]["family_ret1"]),
        ("fh512+dct K4",     data["chained_retrieval"]["prot_t5_xl_fh512_dct_K4"]["family_ret1"]),
        ("rp512+[mean|max]", data["chained_retrieval"]["prot_t5_xl_rp512_mean_max"]["family_ret1"]),
    ]

    # Sort descending
    codecs.sort(key=lambda x: x[1], reverse=True)
    labels = [c[0] for c in codecs]
    vals = np.array([c[1] for c in codecs])
    errs = np.array([normal_ci(v, n) for v in vals])

    # Assign colors by category
    def _color(label):
        if "rp512+" in label or "fh512+" in label:
            return CATEGORY_COLORS["Chained"]
        if label in ("rp512", "fh512"):
            return CATEGORY_COLORS["D-compression"]
        if "dct" in label:
            return CATEGORY_COLORS["Smart pooling"]
        return CATEGORY_COLORS["Pooling-only"]

    colors = [_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, capsize=3, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.7,
                  error_kw=dict(lw=1.2, color="#374151"))

    # Reference lines
    gz = data["uc_retrieval"]["prot_t5_xl_mean_pool"]["family_ret1"]
    ax.axhline(y=gz, color=CODEC_COLORS["ground_zero"], linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.text(len(labels) - 0.5, gz + 0.003, f"ground zero ({gz:.3f})",
            fontsize=8, color=CODEC_COLORS["ground_zero"], ha="right")

    # Trained CC reference (3-seed mean)
    trained_vals = [0.808, 0.785, 0.793]
    trained_mean = np.mean(trained_vals)
    ax.axhline(y=trained_mean, color=CODEC_COLORS["trained_cc"], linestyle=":",
               linewidth=1.5, alpha=0.8)
    ax.text(len(labels) - 0.5, trained_mean + 0.003,
            f"trained CC ({trained_mean:.3f})", fontsize=8,
            color=CODEC_COLORS["trained_cc"], ha="right", fontstyle="italic")

    # Value labels
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + errs[list(vals).index(v)] + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#374151")

    ax.set_ylabel("Family Retrieval Ret@1")
    ax.set_title("Training-Free Codec Benchmark (ProtT5-XL, SCOPe 5K)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0.68, 0.84)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in CATEGORY_COLORS.items()
               if l != "Trained"]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    # Caption
    ax.text(0.5, -0.22, "Error bars: 95% CI, normal approximation (n=850 queries)",
            transform=ax.transAxes, fontsize=8, ha="center", color="#6B7280")

    out = FIG_DIR / "pub_codec_retrieval.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Per-Residue Task Retention
# ---------------------------------------------------------------------------

def fig_per_residue_retention(data):
    """Grouped bars: 4 tasks x 4 methods showing per-residue retention."""
    # Tasks and their data keys
    # All on SCOPe test set except trained CC (CB513)
    tasks = ["SS3 (Q3)", "SS8 (Q8)", "Disorder (rho)", "TM topo (F1)"]

    # Raw baselines (from plm_benchmark_suite on SCOPe test set)
    raw = [
        data["plm_per_residue"]["prot_t5_xl_ss3"]["q3"],
        data["plm_per_residue"]["prot_t5_xl_ss8"]["q8"],
        data["plm_per_residue"]["prot_t5_xl_disorder"]["spearman_rho"],
        data["plm_per_residue"]["prot_t5_xl_tm"]["macro_f1"],
    ]

    # rp512 (from universal_codec_results per_residue)
    rp512 = [
        data["uc_per_residue"]["prot_t5_xl_rp512_ss3"]["q3"],
        data["uc_per_residue"]["prot_t5_xl_rp512_ss8"]["q8"],
        data["uc_per_residue"]["prot_t5_xl_rp512_disorder"]["rho"],
        data["uc_per_residue"]["prot_t5_xl_rp512_tm"]["macro_f1"],
    ]

    # fh512
    fh512 = [
        data["uc_per_residue"]["prot_t5_xl_fh512_ss3"]["q3"],
        data["uc_per_residue"]["prot_t5_xl_fh512_ss8"]["q8"],
        data["uc_per_residue"]["prot_t5_xl_fh512_disorder"]["rho"],
        data["uc_per_residue"]["prot_t5_xl_fh512_tm"]["macro_f1"],
    ]

    # Trained CC d256 (from robust validation, CB513 s42 contrastive)
    # Note: different dataset (CB513) vs SCOPe test set for codecs
    trained_cc = [0.834, 0.692, 0.518, 0.657]

    methods = ["Raw (1024d)", "rp512", "fh512", "Trained CC*"]
    all_vals = [raw, rp512, fh512, trained_cc]
    method_colors = ["#E5E7EB", CODEC_COLORS["rp512"], CODEC_COLORS["fh512"],
                     CODEC_COLORS["trained_cc"]]

    x = np.arange(len(tasks))
    width = 0.18
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (method, vals, color) in enumerate(zip(methods, all_vals, method_colors)):
        edgecolor = "#6B7280" if method == "Raw (1024d)" else "white"
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=method, color=color, edgecolor=edgecolor, linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#374151")

    ax.set_ylabel("Score")
    ax.set_title("Per-Residue Task Retention (ProtT5-XL)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0.4, 0.95)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotations
    ax.text(0.5, -0.15,
            "Per-residue CIs omitted (n>26K residues, all CIs < 0.006). "
            "*Trained CC evaluated on CB513, others on SCOPe test set.",
            transform=ax.transAxes, fontsize=8, ha="center", color="#6B7280")
    ax.text(0.5, -0.20,
            "Chained codecs (rp512+dct_K4) inherit the D-compressor's per-residue performance.",
            transform=ax.transAxes, fontsize=8, ha="center", color="#6B7280",
            fontstyle="italic")

    out = FIG_DIR / "pub_per_residue_retention.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Trade-off Scatter (Retrieval vs Per-Residue)
# ---------------------------------------------------------------------------

def fig_tradeoff_scatter(data):
    """Scatter: x=Ret@1, y=SS3 Q3 for all codecs that have BOTH metrics."""

    # Collect all ProtT5 codecs with both retrieval and SS3 data
    points = []

    # D-compression codecs (have per-residue from Exp 25)
    d_compress = [
        ("rp512", "D-compression",
         data["uc_retrieval"]["prot_t5_xl_rp512"]["family_ret1"],
         data["uc_per_residue"]["prot_t5_xl_rp512_ss3"]["q3"]),
        ("fh512", "D-compression",
         data["uc_retrieval"]["prot_t5_xl_fh512"]["family_ret1"],
         data["uc_per_residue"]["prot_t5_xl_fh512_ss3"]["q3"]),
    ]
    points.extend(d_compress)

    # Pooling-only (mean pool = ground zero, no per-residue since L compressed away)
    # mean_pool has per-residue = raw baseline
    points.append(("mean pool\n(ground zero)", "Pooling-only",
                   data["uc_retrieval"]["prot_t5_xl_mean_pool"]["family_ret1"],
                   data["plm_per_residue"]["prot_t5_xl_ss3"]["q3"]))

    # Smart pooling — dct_K4 pool has per-residue from inverse DCT
    points.append(("dct K4 (inv)", "Smart pooling",
                   data["uc_retrieval"]["prot_t5_xl_dct_K4"]["family_ret1"],
                   data["uc_per_residue"]["prot_t5_xl_dct_K4_inv_ss3"]["q3"]))

    # Chained codecs from Exp 26 — per-residue inherited from D-compressor
    chained_pts = [
        ("rp512+dct K4", "Chained",
         data["chained_retrieval"]["prot_t5_xl_rp512_dct_K4"]["family_ret1"],
         data["uc_per_residue"]["prot_t5_xl_rp512_ss3"]["q3"]),
        ("fh512+dct K4", "Chained",
         data["chained_retrieval"]["prot_t5_xl_fh512_dct_K4"]["family_ret1"],
         data["uc_per_residue"]["prot_t5_xl_fh512_ss3"]["q3"]),
        ("rp512+[m|M]", "Chained",
         data["chained_retrieval"]["prot_t5_xl_rp512_mean_max"]["family_ret1"],
         data["uc_per_residue"]["prot_t5_xl_rp512_ss3"]["q3"]),
    ]
    points.extend(chained_pts)

    # Trained CC (on CB513, different dataset — note in caption)
    points.append(("trained CC*", "Trained",
                   0.795, 0.834))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot by category
    for name, cat, ret1, ss3 in points:
        color = CATEGORY_COLORS[cat]
        marker = "*" if cat == "Trained" else "o"
        size = 200 if cat == "Trained" else 80
        zorder = 10 if cat == "Trained" else 5
        ax.scatter(ret1, ss3, c=color, s=size, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=zorder)
        # Label
        offset_x = 0.005
        offset_y = 0.005
        if "ground zero" in name:
            offset_x = -0.005
            offset_y = -0.012
        elif "trained" in name:
            offset_x = 0.005
            offset_y = -0.015
        ax.annotate(name, (ret1, ss3),
                    xytext=(ret1 + offset_x, ss3 + offset_y),
                    fontsize=7.5, color="#374151")

    # Quadrant reference lines (ground zero)
    gz_ret = data["uc_retrieval"]["prot_t5_xl_mean_pool"]["family_ret1"]
    gz_ss3 = data["plm_per_residue"]["prot_t5_xl_ss3"]["q3"]
    ax.axvline(x=gz_ret, color="#D1D5DB", linestyle="--", linewidth=1)
    ax.axhline(y=gz_ss3, color="#D1D5DB", linestyle="--", linewidth=1)

    # Quadrant labels
    ax.text(0.79, 0.85, "both-task\ngood", fontsize=8, color="#059669",
            fontstyle="italic", ha="center")
    ax.text(0.72, 0.50, "bad at\nboth", fontsize=8, color="#DC2626",
            fontstyle="italic", ha="center")

    ax.set_xlabel("Family Retrieval Ret@1")
    ax.set_ylabel("SS3 Accuracy (Q3)")
    ax.set_title("The Fundamental Trade-off: Retrieval vs Per-Residue (ProtT5-XL)")
    ax.set_xlim(0.70, 0.82)
    ax.set_ylim(0.45, 0.87)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    ax.text(0.5, -0.12,
            "*Trained CC SS3 on CB513; codecs on SCOPe test set",
            transform=ax.transAxes, fontsize=8, ha="center", color="#6B7280")

    out = FIG_DIR / "pub_tradeoff_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Cross-PLM Comparison
# ---------------------------------------------------------------------------

def fig_cross_plm(data):
    """Grouped bars: 3 PLM clusters x 5 top codecs showing PLM >> codec."""
    plms = ["ProtT5-XL", "ESM2-650M", "ESM-C 300M"]
    plm_prefixes = ["prot_t5_xl", "esm2_650m", "esmc_300m"]

    # Top codecs that are available for all PLMs
    codec_defs = [
        ("mean_pool",     "mean_pool",     CATEGORY_COLORS["Pooling-only"]),
        ("[mean|max]",    "mean_max",      "#FBBF24"),
        ("dct K=4",       "dct_K4",        CATEGORY_COLORS["Smart pooling"]),
        ("rp512",         "rp512",         CATEGORY_COLORS["D-compression"]),
        ("fh512",         "fh512",         "#10B981"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    n_plms = len(plms)
    n_codecs = len(codec_defs)
    width = 0.14
    group_width = n_codecs * width + 0.15

    for i, (plm_name, prefix) in enumerate(zip(plms, plm_prefixes)):
        for j, (codec_label, codec_key, color) in enumerate(codec_defs):
            key = f"{prefix}_{codec_key}"
            val = data["uc_retrieval"][key]["family_ret1"]
            xpos = i * group_width + j * width
            err = normal_ci(val, 850)
            bar = ax.bar(xpos, val, width * 0.85, yerr=err, capsize=2,
                         color=color, edgecolor="white", linewidth=0.5,
                         error_kw=dict(lw=0.8, color="#374151"))
            ax.text(xpos, val + err + 0.008, f"{val:.2f}", ha="center",
                    va="bottom", fontsize=6.5, color="#374151")

    # X-axis: PLM group labels
    group_centers = [i * group_width + (n_codecs - 1) * width / 2 for i in range(n_plms)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(plms, fontsize=11, fontweight="bold")

    ax.set_ylabel("Family Retrieval Ret@1")
    ax.set_title("PLM Matters More Than Codec Choice")
    ax.set_ylim(0.0, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bracket annotations for PLM gaps
    ax.annotate("", xy=(group_centers[0], 0.80), xytext=(group_centers[1], 0.80),
                arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=1.5))
    ax.text((group_centers[0] + group_centers[1]) / 2, 0.81, "+0.12",
            ha="center", fontsize=9, color="#DC2626", fontweight="bold")

    # Legend for codecs
    handles = [mpatches.Patch(color=c, label=l) for l, _, c in codec_defs]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9,
              ncol=2)

    ax.text(0.5, -0.10, "Error bars: 95% CI (n=850)",
            transform=ax.transAxes, fontsize=8, ha="center", color="#6B7280")

    out = FIG_DIR / "pub_cross_plm.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Biology & Hierarchy (2x2)
# ---------------------------------------------------------------------------

def fig_biology_hierarchy(data):
    """2x2 multi-panel: EC Ret@1, Pfam Ret@1, GO rho, hierarchy sep ratio."""

    # Top 6 codecs (all ProtT5)
    codec_defs = [
        ("mean pool",      "prot_t5_xl_mean_pool"),
        ("[mean|max]",     "prot_t5_xl_mean_max"),
        ("dct K4",         "prot_t5_xl_dct_K4"),
        ("rp512",          "prot_t5_xl_rp512"),
        ("fh512",          "prot_t5_xl_fh512"),
        ("cos dev",        None),  # From chained
    ]

    colors = [
        CATEGORY_COLORS["Pooling-only"],
        "#FBBF24",
        CATEGORY_COLORS["Smart pooling"],
        CATEGORY_COLORS["D-compression"],
        "#10B981",
        "#60A5FA",
    ]

    # Gather biology data
    bio = data["uc_biology"]
    hier = data["uc_hierarchy"]

    # cosine_deviation biology is in chained results
    cos_dev_row = None
    for row in data["chained_rows"]:
        if row["plm"] == "prot_t5_xl" and row["codec"] == "cosine_deviation":
            cos_dev_row = row
            break

    # Build data arrays for each panel
    ec_vals = []
    pfam_vals = []
    go_vals = []
    sep_vals = []

    for label, key in codec_defs:
        if key is not None:
            ec_vals.append(bio[key]["ec_ret1"])
            pfam_vals.append(bio[key]["pfam_ret1"])
            go_vals.append(bio[key]["go_rho"])
            sep_vals.append(hier[key]["sep_ratio"])
        else:
            # cosine_deviation from chained
            ec_vals.append(cos_dev_row["ec_ret1"])
            pfam_vals.append(cos_dev_row["pfam_ret1"])
            go_vals.append(cos_dev_row["go_rho"])
            sep_vals.append(cos_dev_row["sep_ratio"])

    labels = [cd[0] for cd in codec_defs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    panels = [
        (axes[0, 0], ec_vals,   "EC Number Ret@1",            "EC Enzyme Classification"),
        (axes[0, 1], pfam_vals, "Pfam Family Ret@1",          "Pfam Domain Retrieval"),
        (axes[1, 0], go_vals,   "GO Semantic Similarity (rho)","Gene Ontology Correlation"),
        (axes[1, 1], sep_vals,  "Hierarchy Separation Ratio",  "SCOPe Hierarchy Preservation"),
    ]

    x = np.arange(len(labels))
    for ax, vals, ylabel, title in panels:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5,
                      width=0.65)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#374151")

    fig.suptitle("Biology & Hierarchy Validation (ProtT5-XL)", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()

    out = FIG_DIR / "pub_biology_hierarchy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: Trained ChannelCompressor Addendum
# ---------------------------------------------------------------------------

def fig_trained_addendum(data):
    """Bar chart: trained CC d256/d128 vs original + ground zero codec."""

    # 3-seed trained CC data (seed 42 from Exp 11, seeds 123/456 from Exp 13)
    seed_ret1 = []

    # Seed 42 from channel_compression_results
    for r in data["channel_comp"]:
        if (r.get("name") == "channel_prot_t5_contrastive_d256_s42"
                and "retrieval_family" in r):
            seed_ret1.append(r["retrieval_family"]["precision@1"])
            break

    # Seeds 123, 456 from robust_validation_results
    for r in data["robust_val"]:
        if (r.get("method") == "channel_contrastive"
                and r.get("latent_dim") == 256
                and "retrieval_family" in r):
            seed_ret1.append(r["retrieval_family"]["precision@1"])

    cc_mean = np.mean(seed_ret1)
    cc_std = np.std(seed_ret1)

    # Groups
    labels = [
        "Ground zero\n(mean pool)",
        "Best codec\n(rp512+dct K4)",
        "Trained CC\nd256 (3-seed)",
    ]
    vals = [
        data["uc_retrieval"]["prot_t5_xl_mean_pool"]["family_ret1"],
        data["chained_retrieval"]["prot_t5_xl_rp512_dct_K4"]["family_ret1"],
        cc_mean,
    ]
    errs = [
        normal_ci(vals[0], 850),
        normal_ci(vals[1], 850),
        cc_std,  # Std across 3 seeds
    ]
    bar_colors = [
        CODEC_COLORS["ground_zero"],
        CATEGORY_COLORS["Chained"],
        CODEC_COLORS["trained_cc"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, capsize=5, color=bar_colors,
                  edgecolor="white", linewidth=0.5, width=0.55,
                  error_kw=dict(lw=1.5, color="#374151"))

    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#374151")

    ax.set_ylabel("Family Retrieval Ret@1")
    ax.set_title("Training-Free vs Trained Compression (ProtT5-XL)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.65, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotations
    # Gap between codec and trained
    gap = cc_mean - vals[1]
    ax.annotate("", xy=(1, vals[1] + 0.002), xytext=(2, cc_mean - 0.002),
                arrowprops=dict(arrowstyle="<->", color="#B45309", lw=1.5))
    mid_y = (vals[1] + cc_mean) / 2
    ax.text(1.7, mid_y, f"+{gap:.3f}\n(training\ngain)", fontsize=8,
            color="#B45309", ha="center", fontstyle="italic")

    ax.text(0.5, -0.15,
            "Codec error bars: 95% CI (n=850). Trained CC: +/- 1 std (3 seeds).",
            transform=ax.transAxes, fontsize=8, ha="center", color="#6B7280")

    out = FIG_DIR / "pub_trained_addendum.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 7: Storage Comparison
# ---------------------------------------------------------------------------

def fig_storage_comparison(data):
    """Horizontal bar chart: bytes per protein for each codec/representation."""
    # Mean sequence length from SCOPe 5K dataset
    mean_L = 175  # overall mean; test set ~164

    # ProtT5: D=1024, float32 = 4 bytes/value
    D = 1024
    B = 4  # bytes per float32

    codecs = [
        ("Raw (L x 1024)",       mean_L * D * B),
        ("rp512 (L x 512)",      mean_L * 512 * B),
        ("fh512 (L x 512)",      mean_L * 512 * B),
        ("Trained CC (L x 256)", mean_L * 256 * B),
        ("rp512 + dct K4\n(L x 512) + (2048,)",
                                  mean_L * 512 * B + 2048 * B),
        ("[mean|max] (2048,)",    2048 * B),
        ("mean pool (1024,)",     D * B),
    ]

    labels = [c[0] for c in codecs]
    sizes_kb = [c[1] / 1024 for c in codecs]
    pct_of_raw = [c[1] / codecs[0][1] * 100 for c in codecs]

    # Color by category
    bar_colors = [
        "#E5E7EB",                        # Raw (grey)
        CATEGORY_COLORS["D-compression"], # rp512
        "#10B981",                        # fh512
        CODEC_COLORS["trained_cc"],       # Trained CC
        CATEGORY_COLORS["Chained"],       # rp512+dct_K4
        CATEGORY_COLORS["Pooling-only"],  # [mean|max]
        CODEC_COLORS["ground_zero"],      # mean pool
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    y = np.arange(len(labels))
    bars = ax.barh(y, sizes_kb, color=bar_colors, edgecolor="white",
                   linewidth=0.5, height=0.6)

    # Size + percentage labels
    for bar, kb, pct in zip(bars, sizes_kb, pct_of_raw):
        if kb > 50:
            ax.text(bar.get_width() - 5, bar.get_y() + bar.get_height() / 2,
                    f"{kb:.0f} KB ({pct:.0f}%)", ha="right", va="center",
                    fontsize=9, fontweight="bold", color="white")
        else:
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"{kb:.0f} KB ({pct:.0f}%)", ha="left", va="center",
                    fontsize=9, fontweight="bold", color="#374151")

    ax.set_xlabel("Storage per Protein (KB)")
    ax.set_title("Embedding Storage Comparison (ProtT5-XL, mean L=175)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.5, -0.18,
            "Float32 storage, uncompressed. Gzip typically reduces by 30-50%.",
            transform=ax.transAxes, fontsize=8, ha="center", color="#6B7280")

    out = FIG_DIR / "pub_storage_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading benchmark data...\n")
    data = load_all_data()

    print("Generating publication figures (300 DPI)...\n")
    fig_codec_retrieval(data)
    fig_per_residue_retention(data)
    fig_tradeoff_scatter(data)
    fig_cross_plm(data)
    fig_biology_hierarchy(data)
    fig_trained_addendum(data)
    fig_storage_comparison(data)

    print(f"\nDone! All figures saved to {FIG_DIR}/")
