"""Shared plotting style for all COLM paper figures.

Colorblind-friendly Okabe-Ito palette, clean minimalist axes.
Import and call `apply_style()` before building any figure.
"""
import matplotlib as mpl

# Okabe-Ito palette
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GRAY = "#666666"
LIGHT_GRAY = "#BBBBBB"
BLACK = "#000000"

SINGLE_COL_WIDTH = 6.5  # inches


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def hide_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_both(fig, out_dir, name):
    """Save fig as both vector PDF and 300-dpi PNG."""
    import os
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    png_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    return pdf_path, png_path
