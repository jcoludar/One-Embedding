#!/usr/bin/env python3
"""Plots for Exp 55: VEP retention bars + RNS↔VEP scatter plots.

Figures produced
----------------
docs/figures/exp55_retention.png  — always; per-codec DMS retention bar with CIs
docs/figures/exp55_rns_raw.png    — if RNS data present; raw-codec RNS↔VEP scatter
docs/figures/exp55_rns_binary.png — if binary_896 RNS data present; compressed scatter

JSON schema (Task 11 output)
----------------------------
{
  "experiment": "exp55_vep_retention",
  "codecs_evaluated": [...],
  "smoke_test": bool,
  "dms_retention": {
    "codecs": {
      "<name>": {
        "mean_spearman_rho": float,
        "per_assay_rho": {assay_id: float, ...},
        "retention_pct":   float,
        "retention_ci_low":  float,
        "retention_ci_high": float,
        "n_assays": int
      },
      ...
    },
    "assay_ids": [...]
  },
  "clinvar_auc": {"skipped": bool, "reason": str}
              | {"codecs": {"<name>": {"auc": float, "n": int}, ...}},
  "rns": {"skipped": bool, "reason": str}
       | {
           "raw_rns": [float, ...],
           "binary_896_rns": [float, ...],
           "vep_spearman": [float, ...],
           "pearson_raw": float,
           "pearson_binary_896": float
         },
  "total_time_s": float
}
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "data" / "benchmarks" / "rigorous_v1" / "exp55_vep_retention.json"
FIG_DIR = ROOT / "docs" / "figures"

# ── Colour palette (consistent with project-wide style) ──────────────────────
PAL = {
    "lossless":    "#D5D5D8",   # light grey  — raw / lossless baseline
    "binary_896":  "#7B2040",   # deep maroon — One Embedding binary
    "pq_224":      "#1A9E8F",   # teal        — PQ M=224
    "int4":        "#4A7FBF",   # medium blue — int4
    "fp16":        "#D4903C",   # amber       — fp16
    "default":     "#7B68AE",   # purple      — any other codec
}
EDGE_COLOR = "white"


def _codec_color(name: str) -> str:
    return PAL.get(name, PAL["default"])


def _codec_label(name: str) -> str:
    labels = {
        "lossless":   "Lossless (fp32)",
        "binary_896": "Binary 896d (37×)",
        "pq_224":     "PQ M=224 896d (18×)",
        "int4":       "int4 896d (9×)",
        "fp16":       "fp16 896d (2.3×)",
    }
    return labels.get(name, name)


# ── Figure 1: Retention bar plot ─────────────────────────────────────────────

def plot_retention(payload: dict, out_path: Path) -> None:
    """Per-codec DMS VEP retention bar chart with BCa CI error bars."""
    dms = payload.get("dms_retention")
    if not dms or "codecs" not in dms:
        warnings.warn("dms_retention.codecs not found in JSON — skipping retention plot.")
        return

    codecs_data = dms["codecs"]
    # Sort: lossless first, then by retention_pct descending
    def sort_key(item):
        name, info = item
        if name == "lossless":
            return (0, -info.get("retention_pct", 0))
        return (1, -info.get("retention_pct", 0))

    ordered = sorted(codecs_data.items(), key=sort_key)

    names = [k for k, _ in ordered]
    retentions = [v.get("retention_pct", 0.0) for _, v in ordered]
    ci_lows  = [v.get("retention_ci_low",  r) for (_, v), r in zip(ordered, retentions)]
    ci_highs = [v.get("retention_ci_high", r) for (_, v), r in zip(ordered, retentions)]

    # Error bar magnitudes (asymmetric)
    err_low  = [r - lo for r, lo in zip(retentions, ci_lows)]
    err_high = [hi - r for r, hi in zip(retentions, ci_highs)]

    colors = [_codec_color(n) for n in names]
    labels = [_codec_label(n) for n in names]

    x = np.arange(len(names))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(max(5, 2.2 * len(names)), 5))
    bars = ax.bar(x, retentions, width=bar_w,
                  color=colors, edgecolor=EDGE_COLOR, linewidth=0.8, zorder=2)
    ax.errorbar(x, retentions,
                yerr=[err_low, err_high],
                fmt="none", color="#222222", capsize=5, linewidth=1.4, zorder=3)

    # Annotate bar tops
    for xi, (ret, label_) in enumerate(zip(retentions, labels)):
        ax.text(xi, ret + max(err_high) * 0.15 + 0.3,
                f"{ret:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("VEP Retention (%)", fontsize=11)
    ax.set_title("Exp 55 — DMS VEP Retention per Codec\n(Spearman ρ vs lossless baseline, BCa CIs)",
                 fontsize=11)

    ymin = max(0, min(ci_lows) - 5)
    ymax = min(105, max(ci_highs) + 6)
    ax.set_ylim(ymin, ymax)
    ax.axhline(100, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_assays = list(codecs_data.values())[0].get("n_assays", "?")
    smoke = payload.get("smoke_test", False)
    caption = f"n={n_assays} DMS assay(s)"
    if smoke:
        caption += " [smoke test]"
    ax.text(0.98, 0.02, caption, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#666666")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[55_make_figures] Wrote {out_path}")


# ── Figure 2: RNS ↔ VEP scatter ──────────────────────────────────────────────

def plot_rns_scatter(
    rns_scores: list[float],
    vep_scores: list[float],
    pearson_r: float,
    codec_name: str,
    out_path: Path,
) -> None:
    """Scatter of per-protein RNS vs mean VEP Spearman ρ for one codec."""
    if len(rns_scores) == 0 or len(vep_scores) == 0:
        warnings.warn(f"Empty RNS/VEP arrays for {codec_name} — skipping scatter.")
        return
    if len(rns_scores) != len(vep_scores):
        warnings.warn(
            f"RNS length {len(rns_scores)} != VEP length {len(vep_scores)} "
            f"for {codec_name} — skipping scatter."
        )
        return

    x = np.array(rns_scores, dtype=float)
    y = np.array(vep_scores, dtype=float)

    color = _codec_color(codec_name)
    label = _codec_label(codec_name)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(x, y, c=color, edgecolors="#333333", linewidths=0.5,
               alpha=0.75, s=40, zorder=2)

    # Trend line
    if len(x) >= 3:
        m, b = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 200)
        ax.plot(xfit, m * xfit + b, color="#333333", linewidth=1.2,
                linestyle="--", zorder=3)

    ax.set_xlabel("RNS (Random Neighbor Score)", fontsize=11)
    ax.set_ylabel("VEP Spearman ρ (mean per protein)", fontsize=11)
    ax.set_title(
        f"Exp 55 — RNS vs VEP: {label}\nPearson r = {pearson_r:.3f}",
        fontsize=10,
    )
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[55_make_figures] Wrote {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RESULTS.exists():
        sys.exit(f"[55_make_figures] Results JSON not found: {RESULTS}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(RESULTS.read_text())

    # Figure 1: retention bars (always)
    plot_retention(payload, FIG_DIR / "exp55_retention.png")

    # Figure 2+: RNS scatter (only if RNS section is present and not skipped)
    rns_section = payload.get("rns", {})
    if rns_section.get("skipped", True):
        print(
            "[55_make_figures] RNS section absent or skipped "
            f"({rns_section.get('reason', 'no reason given')}) — "
            "skipping scatter plots."
        )
        return

    # Expected RNS keys (full-run schema)
    raw_rns     = rns_section.get("raw_rns", [])
    binary_rns  = rns_section.get("binary_896_rns", [])
    vep_scores  = rns_section.get("vep_spearman", [])
    pearson_raw    = rns_section.get("pearson_raw", float("nan"))
    pearson_binary = rns_section.get("pearson_binary_896", float("nan"))

    if raw_rns and vep_scores:
        plot_rns_scatter(
            raw_rns, vep_scores, pearson_raw,
            codec_name="lossless",
            out_path=FIG_DIR / "exp55_rns_raw.png",
        )
    else:
        print("[55_make_figures] raw_rns or vep_spearman empty — skipping raw scatter.")

    if binary_rns and vep_scores:
        plot_rns_scatter(
            binary_rns, vep_scores, pearson_binary,
            codec_name="binary_896",
            out_path=FIG_DIR / "exp55_rns_binary.png",
        )
    else:
        print("[55_make_figures] binary_896_rns empty — skipping binary scatter.")

    print(f"[55_make_figures] Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
