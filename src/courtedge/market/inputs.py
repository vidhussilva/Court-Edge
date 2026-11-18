"""Assemble backtest inputs from the interim tables written in Phase 0."""

from __future__ import annotations

import pandas as pd

from courtedge.data.align import point_end_times, score_timeline
from courtedge.market.backtest import MatchInput
from courtedge.markov.score import MatchFormat


def market_frame(betfair_long: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """OPEN ticks for one market with A's and B's last traded prices side by side."""
    g = betfair_long[(betfair_long.market_id == row.market_id)
                     & (betfair_long.market_status == "OPEN")]  # fmt: skip
    wide = g.pivot(index="timestamp_utc", columns="selection_id", values="last_traded_price")
    wide = wide[[int(row.selection_a), int(row.selection_b)]]
    in_play = g.drop_duplicates("timestamp_utc").set_index("timestamp_utc")["in_play"]
    out = pd.DataFrame({"odds_a": wide.iloc[:, 0], "odds_b": wide.iloc[:, 1],
                        "in_play": in_play.reindex(wide.index)})  # fmt: skip
    out = out.dropna(subset=["odds_a", "odds_b"])
    out = out[(out.odds_a > 1) & (out.odds_b > 1)]
    return out.rename_axis("time_utc").reset_index().sort_values("time_utc")


def _rate(row: pd.Series) -> float | None:
    v = row.get("market_base_rate")
    return None if v is None or pd.isna(v) else float(v)


def build_inputs(
    catalogue: pd.DataFrame,
    matches: pd.DataFrame,
    points: pd.DataFrame,
    betfair_long: pd.DataFrame,
    serve_probs: pd.DataFrame,
) -> tuple[list[MatchInput], pd.DataFrame]:
    """Return the MatchInputs and a per-match summary (timing method, skip reasons)."""
    meta = matches.set_index("match_id")
    sp = serve_probs.set_index("match_id")
    inputs, info = [], []
    for _, row in catalogue.iterrows():
        mid = row.match_id
        m = meta.loc[mid]
        if pd.isna(m.winner) or m.parse_issues:
            info.append(dict(match_id=mid, used=False, reason=m.parse_issues or "no winner"))
            continue
        pts = points[points.match_id == mid].sort_values("point_index")
        times, method = point_end_times(pts, row.inplay_start_utc, row.market_end_utc)
        fmt = MatchFormat(int(m.best_of), m.final_set)
        tl = score_timeline(pts, times, fmt)
        mk = market_frame(betfair_long, row)
        inputs.append(MatchInput(mid, mk, tl, float(sp.loc[mid, "pa"]), float(sp.loc[mid, "pb"]),
                                 fmt, int(m.first_server), int(m.winner),
                                 _rate(row)))  # fmt: skip
        info.append(dict(match_id=mid, used=True, reason="", timing=method, ticks=len(mk)))
    return inputs, pd.DataFrame(info)
