#!/usr/bin/env python3
"""Visual demo: AA sequence → ProtT5 embedding → OE codec → DCT K=4 protein vector.

Walks through every numerical transformation the OneEmbeddingCodec applies, on a
real 10-residue slice from AMFR_HUMAN (the smallest WT in the Exp 55 H5).

Output: docs/figures/exp55_transformations_demo.png
        docs/figures/exp55_transformations_demo.pdf  (vector for paper)

CPU-only — does not touch ProtT5 or MPS, so safe to run while the main
experiment is in flight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.fft import dct

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DMS_H5 = ROOT / "data" / "residue_embeddings" / "prot_t5_xl_proteingym_diversity.h5"
OUT_PNG = ROOT / "docs" / "figures" / "exp55_transformations_demo.png"
OUT_PDF = ROOT / "docs" / "figures" / "exp55_transformations_demo.pdf"

ASSAY = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
N_RES = 10  # decapeptide
D_IN = 1024  # ProtT5 native dim
D_OUT = 896  # OE codec target dim
K_DCT = 4  # DCT coefficients per channel for the protein vector

RP_SEED = 42  # deterministic projection


def _heatmap(ax, mat, title, *, cmap="RdBu_r", vmin=None, vmax=None,
             xlabel=None, ylabel="residue", show_cbar=True, aspect="auto"):
    """Render a (n_residues, n_dims) matrix as a heatmap with consistent style."""
    if vmin is None:
        v = float(np.nanmax(np.abs(mat)))
        vmin, vmax = -v, v
    im = ax.imshow(mat, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.set_yticks(range(mat.shape[0]))
    ax.tick_params(axis="both", labelsize=8)
    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
    return im


def main():
    # ── Load real data ────────────────────────────────────────────────────
    with h5py.File(DMS_H5, "r") as f:
        grp = f[ASSAY]
        full_seq = grp.attrs["wt_sequence"]
        wt_full = grp["wt"][:].astype(np.float32)  # (47, 1024)
        # Compute corpus mean from all 15 WT embeddings (concat across residues)
        corpus_residues = []
        for k in f.keys():
            corpus_residues.append(f[k]["wt"][:].astype(np.float32))
        corpus_mean = np.concatenate(corpus_residues, axis=0).mean(axis=0)  # (1024,)

    seq = full_seq[:N_RES]  # "YFQGQLNAMA"
    raw = wt_full[:N_RES]  # (10, 1024)

    # ── Stage 1: centering (subtract corpus mean) ───────────────────────────
    centered = raw - corpus_mean[None, :]  # (10, 1024)

    # ── Stage 2: random orthogonal projection (1024 → 896) ─────────────────
    # Match the codec's universal_transforms.random_orthogonal_project: a
    # deterministic Gaussian projection matrix scaled by 1/sqrt(d_out).
    rng = np.random.default_rng(RP_SEED)
    R = rng.standard_normal((D_IN, D_OUT)).astype(np.float32) / np.sqrt(D_OUT)
    projected = centered @ R  # (10, 896)

    # ── Stage 3: binary quantization (sign) ────────────────────────────────
    binary = np.sign(projected).astype(np.int8)  # (10, 896), values in {-1, +1}

    # ── Stage 4: DCT K=4 along residue axis per channel ────────────────────
    # The codec uses DCT-II along the residue (time) axis with K=4 retained
    # coefficients per channel. We DCT the binary representation here (matching
    # the codec's protein_vec) — first dequantize binary to {-1, +1} which it
    # already is, then DCT.
    dct_full = dct(binary.astype(np.float32), axis=0, type=2, norm="ortho")  # (10, 896)
    dct_k = dct_full[:K_DCT]  # (4, 896)
    protein_vec = dct_k.flatten()  # (3584,)

    # ── Sizes / compression annotations ────────────────────────────────────
    sizes = {
        "raw fp32": raw.nbytes,
        "raw fp16": raw.nbytes // 2,
        "centered": centered.nbytes,
        "projected": projected.nbytes,
        "binary": binary.size // 8,  # 1 bit per element packed
        "protein_vec fp16": protein_vec.size * 2,
    }

    # ──────────────────────────────────────────────────────────────────────
    # ── Render the figure ─────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 28), dpi=110)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        nrows=8, ncols=1, figure=fig,
        height_ratios=[0.6, 1.5, 1.5, 1.4, 1.4, 1.6, 1.4, 0.7],
        hspace=0.55, top=0.97, bottom=0.04, left=0.06, right=0.96,
    )

    # 0: title + sequence strip
    ax_title = fig.add_subplot(gs[0])
    ax_title.set_axis_off()
    ax_title.text(
        0.5, 0.85,
        "OneEmbeddingCodec — sequence → embedding → OE → DCT-K4 protein vector",
        ha="center", va="center", fontsize=18, fontweight="bold",
    )
    ax_title.text(
        0.5, 0.45,
        f"Decapeptide demo: AMFR_HUMAN (residues 1–10), corpus = 15 ProteinGym DMS WTs",
        ha="center", va="center", fontsize=12, color="#555",
    )

    # Sequence as boxes
    ax_seq = fig.add_subplot(gs[0])
    ax_seq.set_xlim(-0.5, N_RES - 0.5)
    ax_seq.set_ylim(-1, 1)
    ax_seq.set_axis_off()
    for i, aa in enumerate(seq):
        ax_seq.add_patch(plt.Rectangle((i - 0.4, -0.7), 0.8, 0.7,
                                        facecolor="#fafafa",
                                        edgecolor="#333", linewidth=0.8))
        ax_seq.text(i, -0.35, aa, ha="center", va="center",
                    fontsize=14, fontweight="bold", family="monospace")
        ax_seq.text(i, -0.95, f"{i+1}", ha="center", va="top", fontsize=8, color="#888")

    # 1: raw ProtT5 embedding
    ax1 = fig.add_subplot(gs[1])
    _heatmap(
        ax1, raw,
        f"① Raw ProtT5 embedding   shape (10, 1024)   "
        f"range [{raw.min():.2f}, {raw.max():.2f}]   "
        f"mean={raw.mean():.3f}   "
        f"size={sizes['raw fp32']/1024:.1f} KB (fp32) / {sizes['raw fp16']/1024:.1f} KB (fp16)",
        xlabel="ProtT5 hidden dimension (0 — 1023)",
    )

    # 2: centered
    ax2 = fig.add_subplot(gs[2])
    _heatmap(
        ax2, centered,
        f"② Centered: x − corpus_mean   shape (10, 1024)   "
        f"range [{centered.min():.2f}, {centered.max():.2f}]   "
        f"mean={centered.mean():.3f}   "
        f"corpus mean computed across all 15 ProteinGym WTs ({sum(c.shape[0] for c in corpus_residues)} residues)",
        xlabel="hidden dimension",
    )

    # 3: projected
    ax3 = fig.add_subplot(gs[3])
    _heatmap(
        ax3, projected,
        f"③ Random projection R ∈ ℝ^(1024×896)   shape (10, 896)   "
        f"range [{projected.min():.2f}, {projected.max():.2f}]   "
        f"R is a fixed Gaussian random matrix, scale 1/√896, seed={RP_SEED}",
        xlabel="projected dimension (0 — 895)",
    )

    # 4: binary
    ax4 = fig.add_subplot(gs[4])
    _heatmap(
        ax4, binary, cmap="gray_r", vmin=-1, vmax=1,
        title=f"④ Binary quantization: sign(·) ∈ {{−1, +1}}   shape (10, 896)   "
              f"size = 10 × 896 bits ÷ 8 = {sizes['binary']} bytes  "
              f"(1.12 KB for 10 residues)",
        xlabel="bit index (0 — 895)",
    )

    # 5: DCT process — show full DCT (10, 896) with the K=4 cutoff highlighted
    ax5 = fig.add_subplot(gs[5])
    v = float(np.nanmax(np.abs(dct_full)))
    im5 = ax5.imshow(dct_full, cmap="RdBu_r", vmin=-v, vmax=v,
                     aspect="auto", interpolation="nearest")
    ax5.axhline(K_DCT - 0.5, color="#ff6b00", linewidth=2.5, linestyle="--")
    ax5.text(896 - 5, K_DCT - 0.5, f" K = {K_DCT}  ↓ keep above ", color="#ff6b00",
             fontsize=11, fontweight="bold", va="center", ha="right")
    ax5.set_title(
        f"⑤ DCT-II along residue axis (per channel)   shape (10, 896) → keep first K=4 rows → (4, 896)   "
        f"orange line marks the K=4 cutoff",
        fontsize=12, loc="left",
    )
    ax5.set_ylabel("DCT coefficient index\n(0 = DC, 1 = AC1, …)", fontsize=9)
    ax5.set_xlabel("channel (0 — 895)", fontsize=9)
    ax5.set_yticks(range(10))
    ax5.tick_params(axis="both", labelsize=8)
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.012, pad=0.01)
    cbar5.ax.tick_params(labelsize=8)

    # 6: protein_vec — show as a 1D bar chart of all 3584 components
    ax6 = fig.add_subplot(gs[6])
    ax6.bar(np.arange(protein_vec.size), protein_vec, width=1.0, color="#3b6db1", linewidth=0)
    ax6.set_title(
        f"⑥ Flatten DCT(4, 896) → protein_vec ∈ ℝ^3584   "
        f"size = 3584 × 2 B (fp16) = {sizes['protein_vec fp16']/1024:.1f} KB   "
        f"this single vector is what gets used for retrieval / clustering / UMAP",
        fontsize=12, loc="left",
    )
    ax6.set_xlim(0, protein_vec.size)
    ax6.set_xlabel("flattened index (0 — 3583)", fontsize=9)
    ax6.set_ylabel("value", fontsize=9)
    ax6.tick_params(axis="both", labelsize=8)
    ax6.axvline(896, color="#999", linewidth=0.5, linestyle=":")
    ax6.axvline(2 * 896, color="#999", linewidth=0.5, linestyle=":")
    ax6.axvline(3 * 896, color="#999", linewidth=0.5, linestyle=":")
    for k in range(4):
        ax6.text(k * 896 + 448, ax6.get_ylim()[1] * 0.92, f"DCT-{k}",
                 ha="center", fontsize=9, color="#555")

    # 7: footer summary
    ax_foot = fig.add_subplot(gs[7])
    ax_foot.set_axis_off()
    summary = (
        f"Per-residue payload: 10 × 896 bits = {sizes['binary']} bytes "
        f"(vs {sizes['raw fp32']} bytes for fp32, {sizes['raw fp16']} bytes for fp16) "
        f"→  {sizes['raw fp32']/sizes['binary']:.0f}× compression on raw fp32, "
        f"{sizes['raw fp16']/sizes['binary']:.0f}× on fp16\n"
        f"Plus a single 3584-d protein vector ({sizes['protein_vec fp16']/1024:.1f} KB) "
        f"for retrieval / clustering / UMAP — this is what appears in benchmarks like SCOPe Ret@1.\n"
        f"Total stored per protein:   "
        f"per-residue ({sizes['binary']} B) + protein_vec ({sizes['protein_vec fp16']} B) "
        f"= {sizes['binary'] + sizes['protein_vec fp16']} bytes for this 10-residue slice "
        f"(7.2 KB at L=10; scales linearly per residue)."
    )
    ax_foot.text(
        0.02, 0.98, summary,
        ha="left", va="top", fontsize=10.5, color="#222",
        family="monospace",
        transform=ax_foot.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fffaf0", edgecolor="#ddd"),
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print()
    print(f"Sequence:        {seq}")
    print(f"Stage shapes:")
    print(f"  ① raw embed:     {raw.shape}    {raw.nbytes/1024:.1f} KB fp32")
    print(f"  ② centered:      {centered.shape}    same size")
    print(f"  ③ projected:     {projected.shape}     {projected.nbytes/1024:.1f} KB fp32")
    print(f"  ④ binary:        {binary.shape}     {sizes['binary']} bytes (1 bit/dim)")
    print(f"  ⑤ DCT (full):    {dct_full.shape}")
    print(f"  ⑤ DCT (K=4):     {dct_k.shape}")
    print(f"  ⑥ protein_vec:   {protein_vec.shape}   {sizes['protein_vec fp16']} bytes fp16")


if __name__ == "__main__":
    main()
