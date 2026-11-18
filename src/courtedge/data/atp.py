"""Pre-match serve-point win probabilities from Sackmann's tennis_atp / tennis_wta files.

For a match between A and B on a surface, as of a date:

1. Each player's serve rate f (points won on serve / serve points) and return rate g
   (points won on return / return points), over the previous 52 weeks.
2. Shrinkage. The overall rates shrink toward the tour average with a prior worth
   ``prior_points`` points. The surface rates then shrink toward the player's own
   overall rate with ``surface_prior_points``. A player with 200 grass points
   gets a grass estimate that is mostly their all-surface rate, which is what we want.
3. Opponent adjustment (Barnett & Clarke, 2005):

       pa = f_t + (f_A - f_t) - (g_B - g_t)

   where f_t is the tour-average serve rate on the surface and g_t = 1 - f_t. A
   strong returner pulls A's serve rate down by however much they beat the average.

No look-ahead. Sackmann dates every match with its tournament's *start* date
(``tourney_date``), so a final played 13 days later has the same date as a first
round. A match is used only if its tournament must have finished before ``as_of``:
``tourney_date + tournament_length < as_of``, with 15 days for slams and 8 days for
everything else. This also excludes the target match's own tournament, so earlier
rounds of the same event are not used. The loss of information is small, and
:func:`test_atp.test_no_future_matches_used` checks the guard.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SLAM_SURFACE = {"ausopen": "Hard", "frenchopen": "Clay", "wimbledon": "Grass", "usopen": "Hard"}


@dataclass(frozen=True)
class ServeModelConfig:
    window_days: int = 364
    prior_points: float = 400.0  # weight of the tour mean in a player's overall rate
    surface_prior_points: float = 300.0  # weight of the overall rate in the surface rate
    clip: tuple[float, float] = (0.40, 0.85)


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


def load_matches(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read and stack atp_matches_YYYY.csv (or wta) files."""
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    return df


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match with serve and return point counts."""
    need = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
    m = matches.dropna(subset=need)
    m = m[(m["w_svpt"] > 0) & (m["l_svpt"] > 0)]
    # Slams last two weeks; everything else at most about one.
    length = np.where(m.get("tourney_level", "") == "G", 15, 8)
    done_by = m["tourney_date"] + pd.to_timedelta(length, unit="D")

    def side(me: str, opp: str) -> pd.DataFrame:
        sv_won = m[f"{me}_1stWon"] + m[f"{me}_2ndWon"]
        opp_won = m[f"{opp}_1stWon"] + m[f"{opp}_2ndWon"]
        return pd.DataFrame(
            {
                "player": m[f"{'winner' if me == 'w' else 'loser'}_name"].map(norm_name),
                "date": m["tourney_date"],
                "done_by": done_by,
                "surface": m["surface"],
                "sv_pts": m[f"{me}_svpt"],
                "sv_won": sv_won,
                "ret_pts": m[f"{opp}_svpt"],
                "ret_won": m[f"{opp}_svpt"] - opp_won,
            }
        )

    return pd.concat([side("w", "l"), side("l", "w")], ignore_index=True)


@dataclass(frozen=True)
class ServeEstimate:
    pa: float
    pb: float
    serve_points_a: int  # points of evidence behind each estimate (all surfaces)
    serve_points_b: int
    tour_serve_rate: float


def _rates(hist: pd.DataFrame, player: str, surface: str, f_t: float, g_t: float,
           cfg: ServeModelConfig) -> tuple[float, float, int]:  # fmt: skip
    h = hist[hist["player"] == player]
    sv_pts, sv_won = h["sv_pts"].sum(), h["sv_won"].sum()
    rt_pts, rt_won = h["ret_pts"].sum(), h["ret_won"].sum()
    f_all = (sv_won + cfg.prior_points * f_t) / (sv_pts + cfg.prior_points)
    g_all = (rt_won + cfg.prior_points * g_t) / (rt_pts + cfg.prior_points)
    s = h[h["surface"] == surface]
    m = cfg.surface_prior_points
    f = (s["sv_won"].sum() + m * f_all) / (s["sv_pts"].sum() + m)
    g = (s["ret_won"].sum() + m * g_all) / (s["ret_pts"].sum() + m)
    return float(f), float(g), int(sv_pts)


def estimate(
    long: pd.DataFrame,
    player_a: str,
    player_b: str,
    as_of: pd.Timestamp,
    surface: str,
    cfg: ServeModelConfig = ServeModelConfig(),
) -> ServeEstimate:
    """Serve-point win probabilities for A and B using only matches finished before as_of."""
    as_of = (
        pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of)
    )
    hist = long[
        (long["done_by"] < as_of) & (long["date"] >= as_of - pd.Timedelta(days=cfg.window_days))
    ]
    on_surface = hist[hist["surface"] == surface]
    base = on_surface if on_surface["sv_pts"].sum() > 10_000 else hist
    f_t = float(base["sv_won"].sum() / base["sv_pts"].sum()) if len(base) else 0.63
    g_t = 1.0 - f_t
    fa, ga, na = _rates(hist, norm_name(player_a), surface, f_t, g_t, cfg)
    fb, gb, nb = _rates(hist, norm_name(player_b), surface, f_t, g_t, cfg)
    lo, hi = cfg.clip
    pa = float(np.clip(f_t + (fa - f_t) - (gb - g_t), lo, hi))
    pb = float(np.clip(f_t + (fb - f_t) - (ga - g_t), lo, hi))
    return ServeEstimate(pa, pb, na, nb, f_t)


def surface_for(row: pd.Series) -> str:
    if "surface" in row and isinstance(row["surface"], str) and row["surface"]:
        return row["surface"]
    return SLAM_SURFACE.get(str(row.get("tournament", "")).lower(), "Hard")


def estimate_catalogue(
    long: pd.DataFrame, catalogue: pd.DataFrame, cfg: ServeModelConfig = ServeModelConfig()
) -> pd.DataFrame:
    """Serve estimates for every catalogue row, as of the market's scheduled start."""
    rows = []
    for _, r in catalogue.iterrows():
        est = estimate(long, r["player_a"], r["player_b"], r["market_time"], surface_for(r), cfg)
        rows.append(dict(match_id=r["match_id"], pa=est.pa, pb=est.pb,
                         serve_points_a=est.serve_points_a, serve_points_b=est.serve_points_b,
                         tour_serve_rate=est.tour_serve_rate))  # fmt: skip
    return pd.DataFrame(rows)
