"""Give every point a clock time (UTC).

Phase 4 aligns points to video. Until then there are two sources of timing:

1. ``elapsed_s`` from ``tennis_slam_pointbypoint`` (its ElapsedTime column). It counts
   from the first serve, which we anchor at the minute the Betfair market turned
   in-play. The anchor error is up to a minute, because BASIC data is minute-level.
2. Otherwise, estimate: spread the points between in-play start and market
   suspension. Each point gets weight 1, a changeover adds 90/40 ≈ 2.25 points and
   a set break adds 3 (from the average point length of about 40 s).

Every report states which method each match used (``timing`` column).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from courtedge.data.pbp import state_at
from courtedge.markov.score import MatchFormat, advance

MEAN_POINT_S = 40.0
CHANGEOVER_S = 90.0
SET_BREAK_S = 120.0


def _elapsed_usable(elapsed: np.ndarray, window_s: float) -> bool:
    if np.isnan(elapsed).any() or len(elapsed) == 0:
        return False
    if np.any(np.diff(elapsed) < 0) or elapsed[-1] <= 0:
        return False
    # Reject clearly broken clocks (zeros everywhere, or far longer than the market).
    return elapsed[-1] <= 1.25 * window_s + 600


def point_end_times(
    points: pd.DataFrame,
    inplay_start: pd.Timestamp,
    market_end: pd.Timestamp,
) -> tuple[pd.Series, str]:
    """Return (end time of each point as UTC timestamps, method used).

    ``points`` holds the rows of one match ordered by ``point_index``.
    """
    window = (market_end - inplay_start).total_seconds()
    elapsed = points["elapsed_s"].to_numpy(dtype=float)
    if _elapsed_usable(elapsed, window):
        secs = elapsed
        method = "elapsed"
    else:
        games = points["games_a"].to_numpy() + points["games_b"].to_numpy()
        sets = points["sets_a"].to_numpy() + points["sets_b"].to_numpy()
        # Weight for the gap *before* each point.
        w = np.ones(len(points))
        new_set = np.r_[False, np.diff(sets) > 0]
        new_game = np.r_[False, np.diff(games) != 0] & ~new_set
        # Changeovers happen after odd-numbered games of a set.
        w[new_game & (games % 2 == 1)] += CHANGEOVER_S / MEAN_POINT_S
        w[new_set] += SET_BREAK_S / MEAN_POINT_S
        secs = np.cumsum(w) / w.sum() * max(window, 1.0)
        method = "estimated"
    times = inplay_start + pd.to_timedelta(secs, unit="s")
    return pd.Series(times, index=points.index), method


def score_timeline(points: pd.DataFrame, end_times: pd.Series, fmt: MatchFormat) -> pd.DataFrame:
    """State of the match *after* each point, stamped with the time it became known.

    Rows: ``time_utc``, ``point_index`` and the score columns of the state after the
    point. A model at time t may use the last row with ``time_utc <= t``.
    """
    rows = []
    for (_, r), t in zip(points.iterrows(), end_times, strict=True):
        after = advance(state_at(r), int(r["winner"]), fmt)
        rows.append(
            dict(
                time_utc=t, point_index=int(r["point_index"]),
                sets_a=after.sets[0], sets_b=after.sets[1],
                games_a=after.games[0], games_b=after.games[1],
                points_a=after.points[0], points_b=after.points[1],
                server=after.server, tiebreak=after.tiebreak,
                tb_first_server=after.tiebreak_first_server,
                match_winner=-1 if after.winner is None else after.winner,
            )
        )  # fmt: skip
    return pd.DataFrame(rows)
