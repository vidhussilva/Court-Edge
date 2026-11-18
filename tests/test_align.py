from pathlib import Path

import numpy as np
import pandas as pd

from courtedge.data import pbp
from courtedge.data.align import point_end_times, score_timeline
from courtedge.markov.score import FinalSet, MatchFormat

FIX = Path(__file__).parent / "fixtures"
T0 = pd.Timestamp("2023-07-16 14:00", tz="UTC")


def _match():
    points, _ = pbp.load_slam(FIX / "slam" / "2023-wimbledon-points.csv")
    return points[points.match_id == "2023-wimbledon-1101"].reset_index(drop=True)


def test_uses_elapsed_when_available():
    pts = _match()
    times, method = point_end_times(pts, T0, T0 + pd.Timedelta(hours=5))
    assert method == "elapsed"
    assert (times - T0).dt.total_seconds().iloc[-1] == pts.elapsed_s.iloc[-1]


def test_estimates_when_elapsed_missing():
    pts = _match().assign(elapsed_s=np.nan)
    end = T0 + pd.Timedelta(hours=4)
    times, method = point_end_times(pts, T0, end)
    assert method == "estimated"
    assert times.is_monotonic_increasing
    assert times.iloc[-1] == end
    gaps = times.diff().dt.total_seconds().dropna()
    assert gaps.max() > 2 * gaps.median()  # set breaks / changeovers are longer


def test_score_timeline_is_state_after_point():
    pts = _match()
    times, _ = point_end_times(pts, T0, T0 + pd.Timedelta(hours=5))
    tl = score_timeline(pts, times, MatchFormat(5, FinalSet.SUPER_TIEBREAK))
    assert len(tl) == len(pts)
    # state after point i == state before point i+1
    nxt = pts.iloc[1:].reset_index(drop=True)
    cur = tl.iloc[:-1].reset_index(drop=True)
    for col in ["sets_a", "sets_b", "games_a", "games_b", "points_a", "points_b", "server"]:
        assert (cur[col].to_numpy() == nxt[col].to_numpy()).all(), col
    assert tl.match_winner.iloc[-1] in (0, 1) and (tl.match_winner.iloc[:-1] == -1).all()
