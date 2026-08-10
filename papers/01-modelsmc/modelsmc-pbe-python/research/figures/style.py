"""Restrained, accessible, deterministic Matplotlib styling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
GREY = "#777777"
LIGHT_GREY = "#E5E7EB"
DARK = "#222222"


@contextmanager
def publication_style() -> Iterator[None]:
    """Apply a journal-safe style without requiring a local LaTeX install."""

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GREY,
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "modelsmc-pbe-publication-figures-v1",
        }
    ):
        yield
