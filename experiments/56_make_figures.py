#!/usr/bin/env python3
"""Plots for Exp 56: VEP codec mega-sweep.

Three figures, sliced by which codec axis they isolate:

  exp56_retention_overview.png   — all arms ranked by retention, BCa CIs
  exp56_abtt_effect.png          — ABTT-k axis on binary 896 (k=0,1,3,8)
  exp56_axes_breakdown.png       — 3-panel: dimensionality / quantization / ABTT×quant

Reads `data/benchmarks/rigorous_v1/exp56_vep_codec_megasweep.json`.
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

RESULTS = ROOT / "data" / "benchmarks" / "rigorous_v1" / "exp56_vep_codec_megasweep.json"
FIG_DIR = ROOT / "docs" / "figures"

# ── Colour palette (extends the Exp 55 palette by codec family) ──────────────
PAL_FAMILY = {
    "lossless":         "#D5D5D8",
    "binary":           "#7B2040",
    "binary_magnitude": "#A53A66",
    "pq":               "#1A9E8F",
    "int4":             "#4A7FBF",
    "int2":             "#2F5180",
    "fp16":             "#D4903C",
}
EDGE = "white"


def _family_of(name: str) -> str:
    if name == "lossless":
        return "lossless"
    if name.startswith("binary_magnitude"):
        return "binary_magnitude"
    if name.startswith("binary"):
        return "binary"
    if name.startswith("pq"):
        return "pq"
    if name.startswith("int4"):
        return "int4"
    if name.startswith("int2"):
        return "int2"
    if name.startswith("fp16"):
        return "fp16"
    return "lossless"


def _color(name: str) -> str:
    return PAL_FAMILY.get(_family_of(name), "#7B68AE")


def _label(name: str) -> str:
    """Human-readable label for plots."""
    pretty = {
        "lossless":             "Lossless 1024d",
        "binary_896_abtt1":     "Binary 896 + ABTT1",
        "binary_896_abtt3":     "Binary 896 + ABTT3",
        "binary_896_abtt8":     "Binary 896 + ABTT8",
        "binary_1024":          "Binary 1024 (no RP)",
        "binary_1024_abtt3":    "Binary 1024 + ABTT3",
        "binary_512":           "Binary 512 (64×)",
        "binary_magnitude_896": "Binary-magnitude 896",
        "pq128_896":            "PQ M=128 896 (32×)",
        "pq64_896":             "PQ M=64 896 (64×)",
        "int2_896":             "int2 896 (18×)",
        "fp16_896_abtt3":       "fp16 896 + ABTT3",
        "int4_896_abtt3":       "int4 896 + ABTT3",
    }
    return pretty.get(name, name)


def _bar(ax, names, retentions, ci_lows, ci_highs, title=None, n_assays=None):
    err_low  = [max(0, r - lo) for r, lo in zip(retentions, ci_lows)]
    err_high = [max(0, hi - r) for r, hi in zip(retentions, ci_highs)]
    colors   = [_color(n) for n in names]
    labels   = [_label(n) for n in names]
    x = np.arange(len(names))

    ax.bar(x, retentions, width=0.6, color=colors, edgecolor=EDGE, linewidth=0.7, zorder=2)
    ax.errorbar(x, retentions, yerr=[err_low, err_high],
                fmt="none", color="#222", capsize=4, linewidth=1.2, zorder=3)
    for xi, r in enumerate(retentions):
        ax.text(xi, r + max(err_high) * 0.15 + 0.3,
                f"{r:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("VEP retention (%)", fontsize=10)
    ymin = max(0, min(ci_lows) - 5) if ci_lows else 0
    ymax = min(105, max(ci_highs) + 6) if ci_highs else 100
    ax.set_ylim(ymin, ymax)
    ax.axhline(100, color="#555", linewidth=0.8, linestyle="--", zorder=1)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    if n_assays is not None:
        ax.text(0.98, 0.02, f"n={n_assays} assays", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#666")


def _extract(payload, names):
    """Pull (retention, ci_low, ci_high) per name from the DMS section, in order.

    Skips names absent from the payload (smoke runs may not include all arms).
    """
    codecs = payload.get("dms_retention", {}).get("codecs", {})
    keep = [n for n in names if n in codecs]
    rets = [codecs[n].get("retention_pct", 0.0) for n in keep]
    los  = [codecs[n].get("retention_ci_low",  r) for n, r in zip(keep, rets)]
    his  = [codecs[n].get("retention_ci_high", r) for n, r in zip(keep, rets)]
    return keep, rets, los, his


def plot_overview(payload, out_path):
    """All arms ordered by retention. Lossless pinned first."""
    codecs = payload.get("dms_retention", {}).get("codecs", {})
    if not codecs:
        warnings.warn("No DMS section in JSON — skipping overview.")
        return
    pairs = sorted(codecs.items(),
                   key=lambda it: (it[0] != "lossless", -it[1].get("retention_pct", 0)))
    names = [k for k, _ in pairs]
    rets  = [v.get("retention_pct", 0) for _, v in pairs]
    los   = [v.get("retention_ci_low",  r) for (_, v), r in zip(pairs, rets)]
    his   = [v.get("retention_ci_high", r) for (_, v), r in zip(pairs, rets)]
    n_assays = list(codecs.values())[0].get("n_assays", "?")

    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(names)), 5.2))
    _bar(ax, names, rets, los, his,
         title="Exp 56 — VEP retention across all codec arms (BCa 95% CI)",
         n_assays=n_assays)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[56_make_figures] Wrote {out_path}")


def plot_abtt_effect(payload, out_path):
    """ABTT-k axis on binary 896 (k=0 from Exp 55 reference, k=1,3,8 here)."""
    # Note: binary_896 (k=0) is in Exp 55, NOT Exp 56. We'll skip it on this plot.
    # If the user runs Exp 56 with binary_896 included, it'll show up.
    names_in_axis = ["binary_896_abtt1", "binary_896_abtt3", "binary_896_abtt8"]
    keep, rets, los, his = _extract(payload, names_in_axis)
    if not keep:
        warnings.warn("No ABTT-k arms found — skipping ABTT panel.")
        return
    n_assays = list(payload["dms_retention"]["codecs"].values())[0].get("n_assays", "?")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _bar(ax, keep, rets, los, his,
         title="Exp 56 — ABTT-k on binary 896d\n(does ABTT hurt VEP like it hurts disorder?)",
         n_assays=n_assays)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[56_make_figures] Wrote {out_path}")


def plot_axes_breakdown(payload, out_path):
    """3-panel grouped view: (a) dimensionality, (b) quantization, (c) ABTT × quant."""
    panels = [
        ("(a) Dimensionality on binary",
         ["lossless", "binary_512", "binary_1024", "binary_1024_abtt3"]),
        ("(b) Quantization variants (896d)",
         ["lossless", "int2_896", "pq64_896", "pq128_896", "binary_magnitude_896"]),
        ("(c) ABTT × quantization (896d)",
         ["lossless", "fp16_896_abtt3", "int4_896_abtt3", "binary_896_abtt3"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    n_assays_ref = "?"
    codecs = payload.get("dms_retention", {}).get("codecs", {})
    if codecs:
        n_assays_ref = list(codecs.values())[0].get("n_assays", "?")

    for ax, (panel_title, names_in_panel) in zip(axes, panels):
        keep, rets, los, his = _extract(payload, names_in_panel)
        if not keep:
            ax.set_visible(False)
            continue
        _bar(ax, keep, rets, los, his, title=panel_title, n_assays=n_assays_ref)

    fig.suptitle("Exp 56 — VEP retention by codec axis", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[56_make_figures] Wrote {out_path}")


def main() -> None:
    if not RESULTS.exists():
        sys.exit(f"[56_make_figures] Results JSON not found: {RESULTS}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(RESULTS.read_text())

    plot_overview(payload, FIG_DIR / "exp56_retention_overview.png")
    plot_abtt_effect(payload, FIG_DIR / "exp56_abtt_effect.png")
    plot_axes_breakdown(payload, FIG_DIR / "exp56_axes_breakdown.png")

    print(f"[56_make_figures] Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
