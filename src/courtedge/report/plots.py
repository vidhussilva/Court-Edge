"""Shared matplotlib style and figures for reports/.

Static PNGs for markdown reports: fixed categorical order (blue, orange, aqua, ...),
2 px lines, recessive grid and axes, text in ink colours rather than series colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_FILL = "#b7d3f6"  # light step of series 1, for confidence bands


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.size": 10,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def probability_trace(
    ax: plt.Axes,
    times,
    series: dict[str, np.ndarray],
    set_marks: list[tuple[object, str]] = (),
    inplay_start=None,
) -> None:
    """Probability lines over time with set boundaries as labelled vertical rules."""
    for i, (label, values) in enumerate(series.items()):
        ax.plot(times, values, color=SERIES[i], label=label)
    for t, text in set_marks:
        ax.axvline(t, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.annotate(text, (t, 1.0), xytext=(3, -2), textcoords="offset points",
                    va="top", fontsize=8, color=INK_2)  # fmt: skip
    if inplay_start is not None:
        ax.axvline(inplay_start, color=INK, linewidth=1.0)
        ax.annotate("in-play", (inplay_start, 0.0), xytext=(3, 2), textcoords="offset points",
                    fontsize=8, color=INK)  # fmt: skip
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=GRID, linewidth=1.0)
    if len(series) > 1:
        ax.legend(loc="lower left")


apply_style()
