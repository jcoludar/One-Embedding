#!/usr/bin/env python3
"""Per-benchmark bar plots for publication.

One figure per metric.  Bar #1 is always the raw ProtT5 baseline (grey),
followed by our codec approaches with consistent colours grouped by
mathematical family:

  Grey         = raw ProtT5 baseline
  Amber        = statistical pooling  ([mean|max])
  Blue         = spectral / frequency  (DCT)
  Teal family  = random linear maps    (RP, FH — both JL-based)
  Purple       = chained codec         (RP → DCT pool)
  Maroon       = One Embedding         (ABTT3 + RP + int4 + DCT)

Generates 7 figures in docs/figures/:
  - 3 retrieval  (family / superfamily / fold  Ret@1)
  - 4 per-residue (SS3 Q3, SS8 Q8, disorder rho, TM F1)
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

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "benchmarks"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (grouped by math family) ───────────────────────────────
PAL = {
    "raw":       "#D5D5D8",   # light grey
    "mean_max":  "#D4903C",   # golden amber   — statistical pooling
    "dct":       "#4A7FBF",   # medium blue    — spectral / frequency
    "rp":        "#1A9E8F",   # teal           — random projection
    "fh":        "#6DD3C3",   # lighter teal   — feature hashing (same family)
    "chained":   "#7B68AE",   # purple         — chained codec
    "oe":        "#7B2040",   # deep maroon    — One Embedding ★
}
EDGE_RAW = "#888888"
EDGE_OTHER = "white"

# Legend grouping
LEGEND_GROUPS = [
    ("Statistical pooling",  PAL["mean_max"]),
    ("Spectral (DCT)",       PAL["dct"]),
    ("Random projection",    PAL["rp"]),
    ("Feature hashing",      PAL["fh"]),
    ("Chained (RP+DCT)",     PAL["chained"]),
    ("One Embedding",        PAL["oe"]),
]

# ── Global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.15,
})


# ── Data loading ──────────────────────────────────────────────────────────
def _load(name):
    with open(DATA / name) as f:
        return json.load(f)


def load_all():
    d = {
        "uc":    _load("universal_codec_results.json"),
        "cc":    _load("chained_codec_results.json"),
        "plm":   _load("plm_benchmark_suite_results.json"),
        "top5":  _load("top5_vs_raw_comparison.json"),
    }
    # Optional: exhaustive sweep + extreme compression (for ALL-methods figs)
    try:
        d["sweep"] = _load("exhaustive_sweep_results.json")
    except FileNotFoundError:
        d["sweep"] = {}
    try:
        d["extreme"] = _load("extreme_compression_results.json")
    except FileNotFoundError:
        d["extreme"] = {}
    # V2 progressive codec results
    try:
        d["v2"] = _load("progressive_codec_results.json")
    except FileNotFoundError:
        d["v2"] = {}
    return d


# ── Extended palette for comprehensive figures ────────────────────────────
CAT_PAL = {
    "one_embedding":  "#7B2040",   # maroon ★
    "preprocessing":  "#C75B7A",   # rose
    "quantization":   "#7CA65C",   # sage green
    "chained":        "#7B68AE",   # purple
    "random_proj":    "#1A9E8F",   # teal
    "feature_hash":   "#6DD3C3",   # light teal
    "holographic":    "#2E7D6C",   # dark teal
    "spectral":       "#4A7FBF",   # blue
    "attention":      "#E07A3A",   # orange
    "pca_based":      "#6B8E9B",   # grey-teal
    "resampling":     "#8FA8C8",   # steel blue
    "pooling":        "#D4903C",   # amber
    "extreme":        "#A0A0A0",   # grey
    "raw_baseline":   "#D5D5D8",   # light grey
}

CAT_LEGEND = [
    ("Raw baseline",      CAT_PAL["raw_baseline"]),
    ("Statistical pool",  CAT_PAL["pooling"]),
    ("Spectral / DCT",    CAT_PAL["spectral"]),
    ("Random projection", CAT_PAL["random_proj"]),
    ("Feature hashing",   CAT_PAL["feature_hash"]),
    ("Holographic (HRR)", CAT_PAL["holographic"]),
    ("Chained codec",     CAT_PAL["chained"]),
    ("Attention / kernel",CAT_PAL["attention"]),
    ("PCA-based",         CAT_PAL["pca_based"]),
    ("Resampling / SVD",  CAT_PAL["resampling"]),
    ("Preprocessing",     CAT_PAL["preprocessing"]),
    ("Quantization",      CAT_PAL["quantization"]),
    ("Extreme compress",  CAT_PAL["extreme"]),
    ("One Embedding",     CAT_PAL["one_embedding"]),
]


def _classify(key, source=""):
    """Assign colour category from method key and source tag."""
    k = key.lower()
    # One Embedding
    if "abtt3_rp512_int4" in k or k == "m5_abtt3_rp512_int4":
        return "one_embedding"
    if "center_abtt3_rp512_int4" in k:
        return "one_embedding"
    # Preprocessing-enhanced chained
    if any(x in k for x in ["abtt", "zscore_abtt", "pca_rot_abtt",
                             "center_abtt"]):
        return "preprocessing"
    if k.startswith("centered_") or k.startswith("zscore_"):
        if any(x in k for x in ["rp512", "dct", "mean"]):
            return "preprocessing"
    # Quantization
    if any(x in k for x in ["int4", "int8", "jpeg", "dpcm", "binary"]):
        return "quantization"
    # Chained codecs
    if source == "cc":
        if any(x in k for x in ["kernel", "attention", "cosine",
                                  "entropy", "svd_spectrum", "token_merge"]):
            return "attention"
        return "chained"
    # Extreme compression
    if source == "extreme":
        return "extreme"
    # Feature hashing
    if "fh" in k and "fh512" in k:
        return "feature_hash"
    # Random projection / sparse
    if any(x in k for x in ["rp512", "rp_", "sparse_rp"]):
        return "random_proj"
    # HRR
    if "hrr" in k:
        return "holographic"
    # Spectral
    if any(x in k for x in ["dct", "haar", "svd_spectrum"]):
        return "spectral"
    # PCA
    if "pca" in k:
        return "pca_based"
    # Kernel / attention
    if any(x in k for x in ["kernel", "attention", "cosine", "entropy"]):
        return "attention"
    # Resampling / SVD / stats
    if any(x in k for x in ["resample", "protein_svd", "channel_stats",
                              "merge", "prune", "cur_k", "nmf",
                              "simhash", "wavelet", "rvq", "pq_", "tt_bd",
                              "ot_"]):
        return "resampling"
    # Baseline
    if k in ("mean_pool", "raw_mean_pool", "prot_t5_xl_mean_pool"):
        return "raw_baseline"
    # Statistical pooling (default for mean/max/percentile etc.)
    return "pooling"


def _prettify(key):
    """Convert raw method key to a readable label."""
    s = key
    s = s.replace("prot_t5_xl_", "")
    # Specific mappings (order matters — longer first)
    for old, new in [
        ("mean_max_euclidean", "[mean|max] (euc)"),
        ("mean_max_euc", "[mean|max] euc"),
        ("mean_max_std", "[mean|max|std]"),
        ("mean_max", "[mean|max]"),
        ("mean_min_max", "[mean|min|max]"),
        ("mean_pool", "mean pool"),
        ("max_pool", "max pool"),
        ("mean_std_concat", "mean+std"),
        ("rp512_dct_K4", "RP512 + DCT K4"),
        ("fh512_dct_K4", "FH512 + DCT K4"),
        ("rp512_mean_max", "RP512 + [mean|max]"),
        ("fh512_mean_max", "FH512 + [mean|max]"),
        ("rp512_resample", "RP512 resample"),
        ("rp512_int4_dct_K4", "RP512 int4 + DCT"),
        ("rp512_int8_dct_K4", "RP512 int8 + DCT"),
        ("channel_resample", "chan. resample"),
        ("channel_stats_", "chan. stats: "),
        ("protein_svd_k", "protein SVD k="),
        ("abtt_k", "ABTT k="),
        ("abtt3", "ABTT3"),
        ("abtt1", "ABTT1"),
        ("center_", "center + "),
        ("zscore_", "zscore + "),
        ("pca_rot_", "PCA-rot + "),
        ("rp512", "RP512"),
        ("fh512", "FH512"),
        ("dct_K4", "DCT K=4"),
        ("dct_K", "DCT K="),
        ("haar_L3", "Haar L3"),
        ("hrr_K", "HRR K="),
        ("svd_spectrum_k64", "SVD spectrum k64"),
        ("svd_spectrum", "SVD spectrum"),
        ("power_mean_p3", "power mean p=3"),
        ("norm_weighted", "norm-weighted"),
        ("kernel_mean_auto", "kernel mean (auto)"),
        ("kernel_mean", "kernel mean"),
        ("cosine_deviation", "cosine deviation"),
        ("attention_to_mean", "attention → mean"),
        ("entropy_weighted", "entropy-weighted"),
        ("token_merge_mean", "token merge"),
        ("trimmed_mean_", "trimmed "),
        ("percentile_p", "percentile "),
        ("mean_iqr", "mean + IQR"),
        ("kmeans_residual_k", "k-means resid. k="),
        ("pca256_", "PCA256 + "),
        ("pca512_", "PCA512 + "),
        ("sparse_rp512", "sparse RP512"),
        ("jpeg_dct_keep", "JPEG DCT keep "),
        ("jpeg_dct50_int4", "JPEG DCT50 + int4"),
        ("delta_order1", "delta order 1"),
        ("dpcm_", "DPCM "),
        ("mean_std_skew", "mean+std+skew"),
    ]:
        s = s.replace(old, new)
    s = s.replace("_", " ").strip()
    return s


# ── Method builders ───────────────────────────────────────────────────────
# Each method is a dict:  label, color, edge, <metric_key>: value

def _m(label, color, **metrics):
    """Shorthand to build a method dict."""
    return {"label": label, "color": color,
            "edge": EDGE_RAW if color == PAL["raw"] else EDGE_OTHER,
            **metrics}


def build_retrieval_methods(d):
    """7 methods for retrieval (family / sf / fold Ret@1)."""
    uc = d["uc"]["results"]["retrieval"]
    cc = d["cc"]["results"]["retrieval"]
    m5 = d["top5"]["M5_abtt3_rp512_int4"]

    def _uc(codec):
        r = uc[f"prot_t5_xl_{codec}"]
        return r["family_ret1"], r["sf_ret1"], r["fold_ret1"]

    def _cc(codec):
        r = cc[f"prot_t5_xl_{codec}"]
        return r["family_ret1"], r["sf_ret1"], r["fold_ret1"]

    methods = []

    # 1  Raw ProtT5 mean-pool baseline
    f, s, fo = _uc("mean_pool")
    methods.append(_m("ProtT5\nraw", PAL["raw"],
                       family=f, sf=s, fold=fo))

    # 2  [mean|max] — statistical pooling
    f, s, fo = _uc("mean_max")
    methods.append(_m("[mean|max]", PAL["mean_max"],
                       family=f, sf=s, fold=fo))

    # 3  DCT K=4 — spectral pooling
    f, s, fo = _uc("dct_K4")
    methods.append(_m("DCT K=4", PAL["dct"],
                       family=f, sf=s, fold=fo))

    # 4  RP 512d — random projection
    f, s, fo = _uc("rp512")
    methods.append(_m("RP 512d", PAL["rp"],
                       family=f, sf=s, fold=fo))

    # 5  FH 512d — feature hashing
    f, s, fo = _uc("fh512")
    methods.append(_m("FH 512d", PAL["fh"],
                       family=f, sf=s, fold=fo))

    # 6  RP + DCT K4 — chained codec
    f, s, fo = _cc("rp512_dct_K4")
    methods.append(_m("RP + DCT", PAL["chained"],
                       family=f, sf=s, fold=fo))

    # 7  One Embedding (ABTT3 + RP + int4 + DCT)  ★
    methods.append(_m("One\nEmbedding", PAL["oe"],
                       family=m5["family_ret1"],
                       sf=m5["superfamily_ret1"],
                       fold=m5["fold_ret1"]))

    return methods


def build_per_residue_methods(d):
    """Methods for per-residue benchmarks (SS3, SS8, disorder, TM).

    The One Embedding has SS3/SS8 data from the top-5 comparison.
    Disorder and TM were not evaluated for ABTT3+RP, so those get
    only 4 bars (raw + DCT-inv + RP + FH).
    """
    plm_pr = d["plm"]["results"]["per_residue"]
    uc_pr  = d["uc"]["results"]["per_residue"]
    m5     = d["top5"]["M5_abtt3_rp512_int4"]

    # 1  Raw ProtT5 (full 1024d)
    raw = _m("ProtT5\nraw", PAL["raw"],
             ss3=plm_pr["prot_t5_xl_ss3"]["q3"],
             ss8=plm_pr["prot_t5_xl_ss8"]["q8"],
             disorder=plm_pr["prot_t5_xl_disorder"]["spearman_rho"],
             tm=plm_pr["prot_t5_xl_tm"]["macro_f1"])

    # 2  DCT K4 inverse reconstruction
    dct = _m("DCT K=4\n(inv)", PAL["dct"],
             ss3=uc_pr["prot_t5_xl_dct_K4_inv_ss3"]["q3"],
             ss8=uc_pr["prot_t5_xl_dct_K4_inv_ss8"]["q8"],
             disorder=uc_pr["prot_t5_xl_dct_K4_inv_disorder"]["rho"],
             tm=uc_pr["prot_t5_xl_dct_K4_inv_tm"]["macro_f1"])

    # 3  RP 512d
    rp = _m("RP 512d", PAL["rp"],
            ss3=uc_pr["prot_t5_xl_rp512_ss3"]["q3"],
            ss8=uc_pr["prot_t5_xl_rp512_ss8"]["q8"],
            disorder=uc_pr["prot_t5_xl_rp512_disorder"]["rho"],
            tm=uc_pr["prot_t5_xl_rp512_tm"]["macro_f1"])

    # 4  FH 512d
    fh = _m("FH 512d", PAL["fh"],
            ss3=uc_pr["prot_t5_xl_fh512_ss3"]["q3"],
            ss8=uc_pr["prot_t5_xl_fh512_ss8"]["q8"],
            disorder=uc_pr["prot_t5_xl_fh512_disorder"]["rho"],
            tm=uc_pr["prot_t5_xl_fh512_tm"]["macro_f1"])

    # 5  One Embedding (ABTT3 + RP512 + int4)
    #    SS3/SS8 from top5 comparison; disorder/TM evaluated separately.
    oe = _m("One\nEmbedding", PAL["oe"],
            ss3=m5["ss3_q3"],
            ss8=m5["ss8_q8"],
            disorder=0.5973,
            tm=0.7518)

    return [raw, dct, rp, fh, oe]


# ── V2 Codec comparison ──────────────────────────────────────────────────

V2_PAL = {
    "full":     "#7B2040",   # maroon (same as OE v1)
    "balanced": "#A83860",   # lighter maroon
    "compact":  "#D05080",   # rose
    "micro":    "#E88098",   # light pink
    "binary":   "#1A9E8F",   # teal (same JL family)
}


def build_v2_comparison(d):
    """Build methods list for V2 codec tier comparison."""
    v2 = d.get("v2", {}).get("D1", {})
    plm_pr = d["plm"]["results"]["per_residue"]

    methods = []

    # Raw ProtT5 baseline
    methods.append(_m("ProtT5\nraw", PAL["raw"],
                       ss3=plm_pr["prot_t5_xl_ss3"]["q3"],
                       ss8=plm_pr["prot_t5_xl_ss8"]["q8"],
                       disorder=plm_pr["prot_t5_xl_disorder"]["spearman_rho"],
                       tm=plm_pr["prot_t5_xl_tm"]["macro_f1"],
                       size_kb=350.0))

    # V2 modes (from progressive_codec_results)
    mode_order = ["micro", "binary", "compact", "balanced", "full"]
    mode_labels = {
        "micro": "V2 micro\n(PQ32)",
        "binary": "V2 binary\n(1-bit)",
        "compact": "V2 compact\n(PQ64)",
        "balanced": "V2 balanced\n(PQ128)",
        "full": "V2 full\n(int4)",
    }

    for mode in mode_order:
        if mode not in v2:
            continue
        r = v2[mode]
        methods.append({
            "label": mode_labels[mode],
            "color": V2_PAL[mode],
            "edge": EDGE_OTHER,
            "ss3": r["ss3_q3"],
            "ss8": r["ss8_q8"],
            "disorder": r.get("disorder_rho"),
            "tm": r.get("tm_f1"),
            "size_kb": r["per_protein_kb"],
            "family": r["family_ret1"],
        })

    return methods


def fig_pareto(d):
    """Size vs quality Pareto scatter for all V2 modes."""
    v2 = d.get("v2", {}).get("D1", {})
    if not v2:
        print("  SKIP Pareto: no V2 data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    mode_order = ["micro", "compact", "binary", "balanced", "full"]
    mode_labels = {
        "micro": "micro (PQ32)",
        "binary": "binary (1-bit)",
        "compact": "compact (PQ64)",
        "balanced": "balanced (PQ128)",
        "full": "full (int4)",
    }

    for ax, (metric, ylabel, title) in zip(axes, [
        ("ss3_q3", "SS3 Q3", "Per-Residue Quality vs Storage"),
        ("family_ret1", "Family Ret@1", "Retrieval Quality vs Storage"),
    ]):
        for mode in mode_order:
            if mode not in v2:
                continue
            r = v2[mode]
            x = r["per_protein_kb"]
            y = r[metric]
            color = V2_PAL[mode]
            ax.scatter(x, y, c=color, s=120, zorder=5,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(mode_labels[mode], (x, y),
                        xytext=(8, -4), textcoords="offset points",
                        fontsize=7.5, color="#374151")

        # Raw baseline reference
        plm_pr = d["plm"]["results"]["per_residue"]
        if metric == "ss3_q3":
            raw_val = plm_pr["prot_t5_xl_ss3"]["q3"]
        else:
            raw_val = 0.734  # raw mean pool
        ax.axhline(y=raw_val, color="#D0D0D0", linestyle="--", linewidth=1)
        ax.text(50, raw_val + 0.003, f"raw ProtT5 ({raw_val:.3f})",
                fontsize=7, color="#888888")

        ax.set_xlabel("Per-Protein Size (KB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("One Embedding V2: Size–Quality Tradeoff", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "pub_v2_pareto.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Plotting ──────────────────────────────────────────────────────────────

def _normal_ci(p, n=850, z=1.96):
    """95 % CI half-width for a proportion (normal approximation)."""
    return z * np.sqrt(p * (1 - p) / n)


def bar_plot(methods, metric_key, ylabel, title, filename,
             show_ci=False, n_queries=850, fmt=".3f",
             caption=None):
    """Draw and save a single bar chart.

    Args:
        methods: list of method dicts (must have ``label``, ``color``,
                 ``edge``, and *metric_key* as keys).
        metric_key: which value to plot from each method dict.
        ylabel: y-axis label.
        title: figure title.
        filename: output filename (placed in FIG_DIR).
        show_ci: if True, add 95 % CI error bars (retrieval only).
        n_queries: sample size for CI computation.
        fmt: format string for bar value labels.
        caption: optional footnote text below the x-axis.
    """
    # Filter to methods that have data for this metric
    ms = [m for m in methods if metric_key in m]
    if not ms:
        print(f"  SKIP {filename}: no data for '{metric_key}'")
        return

    labels = [m["label"] for m in ms]
    vals   = np.array([m[metric_key] for m in ms])
    colors = [m["color"] for m in ms]
    edges  = [m["edge"] for m in ms]
    errs   = np.array([_normal_ci(v, n_queries) for v in vals]) if show_ci else None

    # Dynamic y limits
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin
    pad = max(span * 0.35, 0.02)            # room above bars for labels
    floor = vmin - max(span * 0.5, 0.015)   # room below smallest bar
    ceil  = vmax + pad
    if errs is not None:
        ceil = (vals + errs).max() + pad * 0.7

    fig, ax = plt.subplots(figsize=(max(7, len(ms) * 1.1), 5))
    x = np.arange(len(ms))

    bar_kw = dict(width=0.62, linewidth=0.8,
                  error_kw=dict(lw=1.2, color="#374151", capsize=4))
    if errs is not None:
        bars = ax.bar(x, vals, yerr=errs, color=colors,
                      edgecolor=edges, **bar_kw)
    else:
        bars = ax.bar(x, vals, color=colors, edgecolor=edges, **bar_kw)

    # Value labels on top of each bar
    for i, (bar, v) in enumerate(zip(bars, vals)):
        y_top = v + (errs[i] if errs is not None else 0) + pad * 0.08
        is_oe = (ms[i]["color"] == PAL["oe"])
        ax.text(bar.get_x() + bar.get_width() / 2, y_top,
                f"{v:{fmt}}", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if is_oe else "normal",
                color=PAL["oe"] if is_oe else "#374151")

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(floor, ceil)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

    # Legend (right side, compact)
    handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=l)
               for l, c in LEGEND_GROUPS
               if any(m["color"] == c for m in ms)]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7.5,
                  framealpha=0.9, handlelength=1.2, handleheight=0.9)

    # Footnote / caption
    if caption:
        ax.text(0.5, -0.15, caption, transform=ax.transAxes,
                fontsize=7.5, ha="center", color="#6B7280")

    out = FIG_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Comprehensive method collection ───────────────────────────────────────

def _add(collected, key, source, family_ret1, ss3_q3=None):
    """Add a method to the collected dict (dedup by label)."""
    label = _prettify(key)
    if label in collected:
        return  # first entry wins (prefer uc/cc over sweep)
    cat = _classify(key, source)
    entry = {"label": label, "color": CAT_PAL[cat],
             "edge": EDGE_RAW if cat == "raw_baseline" else EDGE_OTHER,
             "family": family_ret1}
    if ss3_q3 is not None:
        entry["ss3"] = ss3_q3
    collected[label] = entry


def build_all_retrieval(d):
    """Collect ALL methods that have family_ret1 (ProtT5 only)."""
    collected = {}

    # 1. Universal codec (14 codecs)
    uc_r = d["uc"]["results"]["retrieval"]
    for key, val in uc_r.items():
        if not key.startswith("prot_t5_xl_"):
            continue
        short = key.replace("prot_t5_xl_", "")
        ss3_key = f"{key.replace('_mean_max_euclidean', '_mean_max').replace('_euclidean', '')}_ss3"
        # Try matching per-residue
        uc_pr = d["uc"]["results"].get("per_residue", {})
        ss3 = None
        for pr_key in [f"prot_t5_xl_{short}_ss3",
                       f"prot_t5_xl_{short.replace('_euclidean', '')}_ss3"]:
            if pr_key in uc_pr:
                ss3 = uc_pr[pr_key].get("q3")
                break
        _add(collected, short, "uc", val["family_ret1"], ss3)

    # 2. Chained codecs (11 codecs)
    cc_r = d["cc"]["results"]["retrieval"]
    cc_pr = d["cc"]["results"].get("per_residue", {})
    for key, val in cc_r.items():
        if not key.startswith("prot_t5_xl_"):
            continue
        short = key.replace("prot_t5_xl_", "")
        # Chained per-residue comes from summary_rows
        ss3 = None
        for row in d["cc"]["results"].get("summary_rows", []):
            if row.get("plm") == "prot_t5_xl" and row.get("codec") == short:
                ss3 = row.get("ss3_q3")
                break
        _add(collected, short, "cc", val["family_ret1"], ss3)

    # 3. Exhaustive sweep (parts A, B, C, E, H, J)
    sweep = d.get("sweep", {})
    for part in ["part_A", "part_B", "part_C", "part_E", "part_H", "part_J"]:
        section = sweep.get(part, {})
        for key, val in section.items():
            if not isinstance(val, dict) or "family_ret1" not in val:
                continue
            # Skip ESM2 methods
            if key.startswith("esm2_"):
                continue
            _add(collected, key, "sweep", val["family_ret1"],
                 val.get("ss3_q3"))

    # 4. Top-5 comparison
    for key, val in d.get("top5", {}).items():
        if not isinstance(val, dict) or "family_ret1" not in val:
            continue
        _add(collected, key, "top5", val["family_ret1"],
             val.get("ss3_q3"))

    # 5. Extreme compression (ProtT5 only)
    extreme = d.get("extreme", {})
    pt5_ext = extreme.get("results", {}).get("prot_t5_xl", {})
    for key, val in pt5_ext.items():
        if not isinstance(val, dict) or "family_ret1" not in val:
            continue
        _add(collected, key, "extreme", val["family_ret1"],
             val.get("ss3_q3"))

    # Raw ProtT5 baseline from plm suite
    plm_r = d["plm"]["results"]["retrieval"]
    if "prot_t5_xl_family_cosine" in plm_r:
        plm_pr = d["plm"]["results"]["per_residue"]
        _add(collected, "raw_mean_pool", "plm",
             plm_r["prot_t5_xl_family_cosine"]["ret1"],
             plm_pr.get("prot_t5_xl_ss3", {}).get("q3"))

    return list(collected.values())


# ── Horizontal bar chart for comprehensive figures ────────────────────────

def hbar_plot(methods, metric_key, xlabel, title, filename,
              fmt=".3f", caption=None):
    """Horizontal bar chart with ALL methods, sorted by value."""
    ms = [m for m in methods if metric_key in m]
    if not ms:
        print(f"  SKIP {filename}: no data for '{metric_key}'")
        return

    # Sort by value (ascending → best at top of figure)
    ms.sort(key=lambda m: m[metric_key])

    labels = [m["label"] for m in ms]
    vals   = np.array([m[metric_key] for m in ms])
    colors = [m["color"] for m in ms]
    edges  = [m["edge"] for m in ms]

    n = len(ms)
    fig_h = max(5, n * 0.32 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(n)

    bars = ax.barh(y, vals, height=0.72, color=colors,
                   edgecolor=edges, linewidth=0.5)

    # Value labels at end of each bar
    vmax = vals.max()
    for i, (bar, v) in enumerate(zip(bars, vals)):
        is_oe = (ms[i]["color"] == CAT_PAL["one_embedding"])
        ax.text(v + vmax * 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:{fmt}}", ha="left", va="center",
                fontsize=7, fontweight="bold" if is_oe else "normal",
                color=CAT_PAL["one_embedding"] if is_oe else "#374151")

    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=12, fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, vmax * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Raw baseline reference line
    raw_val = next((m[metric_key] for m in ms
                    if m["color"] == CAT_PAL["raw_baseline"]), None)
    if raw_val is not None:
        ax.axvline(x=raw_val, color="#888888", linestyle="--",
                   linewidth=1, alpha=0.6, zorder=0)
        ax.text(raw_val, n + 0.3, f"raw baseline ({raw_val:{fmt}})",
                fontsize=7, color="#888888", ha="center")

    # Legend (compact, outside bottom)
    used_colors = set(m["color"] for m in ms)
    handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=l)
               for l, c in CAT_LEGEND if c in used_colors]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=6.5,
                  framealpha=0.9, ncol=2, handlelength=1.0, handleheight=0.8)

    if caption:
        ax.text(0.5, -0.04, caption, transform=ax.transAxes,
                fontsize=7, ha="center", color="#6B7280")

    out = FIG_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}  ({n} methods)")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading benchmark data ...")
    d = load_all()

    ret_methods = build_retrieval_methods(d)
    pr_methods  = build_per_residue_methods(d)

    ci_note = "Error bars: 95 % CI (normal approx., n = 850 queries). " \
              "RP/FH seed variance ~ 0.004 (10 seeds)."
    pr_note = "ProtT5 raw = full (L, 1024) embeddings.  " \
              "Codecs compress to (L, 512).  Linear probe on SCOPe test set."

    # ── Curated main figures (7) ──────────────────────────────────────────
    print("\nGenerating curated retrieval bar plots ...")
    bar_plot(ret_methods, "family",  "Ret@1",  "Family Retrieval (ProtT5-XL, SCOPe 5 K)",
             "pub_bar_family_ret1.png", show_ci=True, caption=ci_note)
    bar_plot(ret_methods, "sf",      "Ret@1",  "Superfamily Retrieval (ProtT5-XL, SCOPe 5 K)",
             "pub_bar_sf_ret1.png", show_ci=True, caption=ci_note)
    bar_plot(ret_methods, "fold",    "Ret@1",  "Fold Retrieval (ProtT5-XL, SCOPe 5 K)",
             "pub_bar_fold_ret1.png", show_ci=True, caption=ci_note)

    print("\nGenerating curated per-residue bar plots ...")
    bar_plot(pr_methods, "ss3",      "Q3 Accuracy",
             "Secondary Structure 3-Class (ProtT5-XL)",
             "pub_bar_ss3_q3.png", caption=pr_note)
    bar_plot(pr_methods, "ss8",      "Q8 Accuracy",
             "Secondary Structure 8-Class (ProtT5-XL)",
             "pub_bar_ss8_q8.png", caption=pr_note)
    bar_plot(pr_methods, "disorder", "Spearman ρ",
             "Disorder Prediction (ProtT5-XL, CheZOD)",
             "pub_bar_disorder_rho.png", caption=pr_note)
    bar_plot(pr_methods, "tm",       "Macro F1",
             "Transmembrane Topology (ProtT5-XL, TMbed)",
             "pub_bar_tm_f1.png", caption=pr_note)

    # ── Comprehensive ALL-methods figures ─────────────────────────────────
    print("\nCollecting ALL methods ...")
    all_methods = build_all_retrieval(d)
    print(f"  Found {len(all_methods)} unique methods")

    print("\nGenerating comprehensive ALL-methods figures ...")
    hbar_plot(all_methods, "family", "Family Ret@1",
              "ALL Approaches — Family Retrieval (ProtT5-XL, SCOPe 5 K)",
              "pub_all_family_ret1.png",
              caption="Sorted by Ret@1. Dashed line = raw ProtT5 mean-pool baseline.")
    hbar_plot(all_methods, "ss3", "SS3 Q3 Accuracy",
              "ALL Approaches — Secondary Structure Retention (ProtT5-XL)",
              "pub_all_ss3_q3.png",
              caption="Sorted by Q3. Methods without per-residue data omitted.")

    # ── V2 Codec tier comparison figures ─────────────────────────────────
    if d.get("v2", {}).get("D1"):
        print("\nGenerating V2 codec comparison figures ...")
        v2_methods = build_v2_comparison(d)
        v2_note = "V2 codec modes on ABTT3+RP512.  Codebook fitted on SCOPe train set."

        bar_plot(v2_methods, "ss3", "Q3 Accuracy",
                 "V2 Codec Tiers — SS3 (ProtT5-XL)",
                 "pub_v2_ss3.png", caption=v2_note)
        bar_plot(v2_methods, "disorder", "Spearman ρ",
                 "V2 Codec Tiers — Disorder (ProtT5-XL)",
                 "pub_v2_disorder.png", caption=v2_note)
        bar_plot(v2_methods, "tm", "Macro F1",
                 "V2 Codec Tiers — TM Topology (ProtT5-XL)",
                 "pub_v2_tm.png", caption=v2_note)
        fig_pareto(d)
    else:
        print("\nNo V2 results found, skipping V2 figures")

    print(f"\nDone — all figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
