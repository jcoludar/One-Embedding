"""Shared style for lab-talk figures.

Inspired by SpeciesEmbedding's `tools/visualization/styling.py` conventions
(publication DPI, generous label sizes, restrained palette). Codec-colored:
lossless = cool blue, compressed = warm amber, default = bold green to draw
the eye to the recommended config.
"""

import matplotlib.pyplot as plt

DPI = 200
FS_TITLE = 13
FS_LABEL = 11
FS_TICK = 10
FS_ANNOT = 9

# Codec config palette (bold for default + max-quality; muted for the rest)
CODEC_COLORS = {
    "lossless": "#1f77b4",          # cool blue
    "fp16":     "#5DADE2",
    "int4":     "#F39C12",          # amber
    "pq_max":   "#E67E22",          # darker amber (PQ M=224, recommended max-quality)
    "pq_aggressive": "#922B21",     # deep red (more aggressive PQ)
    "binary":   "#27AE60",          # green (default)
}

# 5-PLM palette (cool→warm gradient, alphabetical by family)
PLM_COLORS = {
    "ProtT5-XL":   "#2E86AB",
    "ProstT5":     "#A23B72",
    "ESM2-650M":   "#F18F01",
    "ESM-C 600M":  "#C73E1D",
    "ANKH-large":  "#3B1F2B",
}

# Diverging colormap centered at 100% retention
RETENTION_CMAP = "RdBu"


def apply_defaults():
    """Apply matplotlib rcParams once per script."""
    plt.rcParams.update({
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "axes.titlesize":     FS_TITLE,
        "axes.labelsize":     FS_LABEL,
        "xtick.labelsize":    FS_TICK,
        "ytick.labelsize":    FS_TICK,
        "legend.fontsize":    FS_TICK,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        "font.family":        "sans-serif",
    })
