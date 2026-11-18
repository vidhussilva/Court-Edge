"""Plot the de-vigged market probability for one match, with set scores marked.

    uv run python scripts/plot_market_trace.py --market-id 1.216000001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from courtedge.data.align import point_end_times, score_timeline
from courtedge.data.catalogue import load_catalogue
from courtedge.market.devig import devig
from courtedge.markov.score import MatchFormat
from courtedge.report import plots


def market_series(bf: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Timestamps with P(A) from de-vigged last traded prices (before settlement)."""
    g = bf[(bf.market_id == row.market_id) & (bf.market_status == "OPEN")]
    wide = g.pivot(index="timestamp_utc", columns="selection_id", values="last_traded_price")
    wide = wide[[int(row.selection_a), int(row.selection_b)]].dropna()
    return pd.DataFrame({"p_a": devig(wide.iloc[:, 0], wide.iloc[:, 1])}, index=wide.index)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--catalogue", default="data/catalogue.csv")
    ap.add_argument("--market-id", help="defaults to the first catalogue row")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--title-prefix", default="")
    args = ap.parse_args(argv)

    interim = Path(args.interim)
    cat = load_catalogue(args.catalogue)
    row = cat.iloc[0] if args.market_id is None else cat[cat.market_id == args.market_id].iloc[0]
    bf = pd.read_parquet(interim / "betfair.parquet")
    pts = pd.read_parquet(interim / "points.parquet")
    pts = pts[pts.match_id == row.match_id].sort_values("point_index")

    mkt = market_series(bf, row)
    times, method = point_end_times(pts, row.inplay_start_utc, row.market_end_utc)
    tl = score_timeline(pts, times, MatchFormat(int(row.best_of), row.final_set))
    set_ends = tl[(tl.sets_a + tl.sets_b).diff().fillna(0) > 0]
    marks = [(r.time_utc, f"{r.sets_a}-{r.sets_b}") for r in set_ends.itertuples()]

    fig, ax = plt.subplots(figsize=(9, 4))
    plots.probability_trace(ax, mkt.index, {"Market P(A wins), de-vigged": mkt.p_a.to_numpy()},
                            marks, row.inplay_start_utc)  # fmt: skip
    ax.set_title(f"{args.title_prefix}{row.player_a} v {row.player_b}: market win probability")
    ax.set_ylabel(f"P({row.player_a.split()[-1]} wins)")
    ax.set_xlabel(f"UTC (set boundaries from {method} point times)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    return plots.save(fig, Path(args.out) / f"market_trace_{row.market_id}.png")


if __name__ == "__main__":
    print(main())
