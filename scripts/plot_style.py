#!/usr/bin/env python3
from __future__ import annotations

"""Unified plotting style for publication-ready figures (BAE / Origin-inspired)."""

import matplotlib as mpl
import seaborn as sns

PUBLICATION_PALETTE = [
    "#2F6DA3",  # primary publication blue
    "#E6862A",  # primary publication orange
    "#78A9D4",  # support blue
    "#F2B57D",  # support orange
    "#2A9D8F",  # teal accent
    "#8A94A6",  # neutral slate
]


def apply_bae_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    sns.set_palette(PUBLICATION_PALETTE)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "Arial Unicode MS", "Arial", "DejaVu Sans", "Liberation Sans", "Noto Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.6,
        "figure.titlesize": 11,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.50,
        "grid.alpha": 0.20,
        "grid.color": "#D9DEE5",
        "lines.linewidth": 1.5,
        "lines.markersize": 5.0,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#C7CDD4",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def get_publication_palette(n: int | None = None) -> list[str]:
    if n is None or n <= len(PUBLICATION_PALETTE):
        return PUBLICATION_PALETTE[:n] if n else PUBLICATION_PALETTE.copy()
    out = []
    while len(out) < n:
        out.extend(PUBLICATION_PALETTE)
    return out[:n]
