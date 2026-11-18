"""Turn Betfair decimal odds into probabilities.

The raw implied probability of a runner is 1 / odds. For a two-runner market the
implied probabilities usually sum to slightly more than 1: that excess is the
overround. Back prices sit above the "true" probability and lay prices below, and
last-traded prices bounce between the two. Every comparison against a model must
remove the overround first ("de-vig"), or the market looks worse calibrated than
it is.

Two methods:

* multiplicative (default): divide each implied probability by their sum. Simple
  and standard for exchange prices, where the overround is small and roughly
  symmetric.
* power: find k with sum(implied_i ** k) = 1. This puts more of the overround on
  the longshot, which matches the favourite-longshot bias seen in bookmaker
  prices. Included for the sensitivity check in the report.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def implied(odds: ArrayLike) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if np.any(o[~np.isnan(o)] <= 1.0):
        raise ValueError("decimal odds must be > 1")
    return 1.0 / o


def overround(odds_a: ArrayLike, odds_b: ArrayLike) -> np.ndarray:
    """Sum of implied probabilities minus one (can be negative on an exchange)."""
    return implied(odds_a) + implied(odds_b) - 1.0


def devig_multiplicative(odds_a: ArrayLike, odds_b: ArrayLike) -> np.ndarray:
    """P(A wins) from the two runners' odds, normalised so P(A) + P(B) = 1."""
    ia, ib = implied(odds_a), implied(odds_b)
    return ia / (ia + ib)


def devig_power(odds_a: ArrayLike, odds_b: ArrayLike, tol: float = 1e-12) -> np.ndarray:
    """P(A wins) using the power method: solve ia**k + ib**k = 1 for k by bisection."""
    ia = np.atleast_1d(implied(odds_a))
    ib = np.atleast_1d(implied(odds_b))
    lo = np.full(ia.shape, 0.2)
    hi = np.full(ia.shape, 5.0)
    for _ in range(200):  # bisection: the sum falls as k rises, since 0 < i < 1
        mid = (lo + hi) / 2
        too_big = ia**mid + ib**mid > 1
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
        if np.all(hi - lo < tol):
            break
    k = (lo + hi) / 2
    out = ia**k
    return out if np.ndim(odds_a) else out[0]


def devig(odds_a: ArrayLike, odds_b: ArrayLike, method: str = "multiplicative") -> np.ndarray:
    if method == "multiplicative":
        return devig_multiplicative(odds_a, odds_b)
    if method == "power":
        return devig_power(odds_a, odds_b)
    raise ValueError(f"unknown de-vig method {method!r}")
