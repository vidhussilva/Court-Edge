"""Betfair price ladder: valid prices run from 1.01 to 1000 with band-specific increments."""

from __future__ import annotations

# (upper edge of band, increment inside it). The band is (previous edge, upper edge].
LADDER = [(2, 0.01), (3, 0.02), (4, 0.05), (6, 0.1), (10, 0.2), (20, 0.5), (30, 1),
          (50, 2), (100, 5), (1000, 10)]  # fmt: skip


def to_tick(price: float) -> float:
    """Round a decimal price to the nearest valid tick, clamped to [1.01, 1000]."""
    price = min(max(price, 1.01), 1000.0)
    lo = 1.0
    for hi, step in LADDER:
        if price <= hi:
            return round(lo + round((price - lo) / step) * step, 2)
        lo = hi
    return 1000.0


def tick_down(price: float, ticks: int = 1) -> float:
    """Move a valid price down the ladder (a worse price for a backer)."""
    p = to_tick(price)
    for _ in range(ticks):
        if p <= 1.01:
            return 1.01
        # A valid tick inside (lo, hi] is at least one step above lo, so one step down
        # stays on the ladder.
        step = next(step for hi, step in LADDER if p <= hi + 1e-9)
        p = round(p - step, 2)
    return p
