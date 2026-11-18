"""Forecast scoring and bootstrap intervals.

In-play forecasts from the same match are strongly correlated (one outcome is
scored hundreds of times), so every confidence interval resamples whole matches,
never individual ticks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

_EPS = 1e-6


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean forecast vs observed frequency in equal-width probability bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append(dict(bin_lo=edges[b], bin_hi=edges[b + 1], n=int(mask.sum()),
                             mean_p=float(p[mask].mean()), freq=float(y[mask].mean())))  # fmt: skip
    return pd.DataFrame(rows)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    t = calibration_table(p, y, bins)
    return float((t["n"] * (t["mean_p"] - t["freq"]).abs()).sum() / t["n"].sum())


def bootstrap_by_group(
    df: pd.DataFrame,
    group: str,
    stat: Callable[[pd.DataFrame], float],
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) with whole groups (matches) resampled."""
    point = stat(df)
    groups = df[group].unique()
    if len(groups) < 2:
        return point, np.nan, np.nan
    parts = {g: d for g, d in df.groupby(group, sort=False)}
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for i in range(n):
        pick = rng.choice(groups, size=len(groups), replace=True)
        draws[i] = stat(pd.concat([parts[g] for g in pick], ignore_index=True))
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
