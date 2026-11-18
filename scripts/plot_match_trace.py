"""Replay one catalogue match point by point and plot P(player A wins the match).

    uv run python scripts/plot_match_trace.py --match-id 2023-wimbledon-1101 --pa 0.66 --pb 0.64

Serve-point probabilities default to the Phase 2 pre-match estimates when
data/interim/serve_probs.parquet exists; otherwise pass --pa/--pb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from courtedge.data.pbp import state_at
from courtedge.markov.engine import win_probabilities
from courtedge.markov.score import MatchFormat, advance
from courtedge.report import plots


def trace(points: pd.DataFrame, pa: float, pb: float, fmt: MatchFormat) -> pd.DataFrame:
    """P(A wins game/set/match) before the first point and after every point."""
    rows = []
    state = state_at(points.iloc[0])
    w = win_probabilities(state, pa, pb, fmt)
    rows.append(dict(point=0, sets_a=0, sets_b=0, game=w.game, set=w.set, match=w.match))
    for i, (_, r) in enumerate(points.iterrows(), start=1):
        state = advance(state_at(r), int(r["winner"]), fmt)
        w = win_probabilities(state, pa, pb, fmt)
        rows.append(dict(point=i, sets_a=state.sets[0], sets_b=state.sets[1],
                         game=w.game, set=w.set, match=w.match))  # fmt: skip
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--match-id", help="defaults to the first match in matches.parquet")
    ap.add_argument("--pa", type=float, help="P(A wins a point on serve)")
    ap.add_argument("--pb", type=float, help="P(B wins a point on serve)")
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--title-prefix", default="")
    args = ap.parse_args(argv)

    interim = Path(args.interim)
    matches = pd.read_parquet(interim / "matches.parquet")
    m = (
        matches.iloc[0]
        if args.match_id is None
        else matches.set_index("match_id").loc[args.match_id].rename(args.match_id)
    )
    match_id = m.name if args.match_id else m.match_id
    pa, pb = args.pa, args.pb
    sp = interim / "serve_probs.parquet"
    if (pa is None or pb is None) and sp.exists():
        est = pd.read_parquet(sp).set_index("match_id").loc[match_id]
        pa, pb = est.pa, est.pb
    if pa is None or pb is None:
        ap.error("no serve probabilities: pass --pa and --pb or run the Phase 2 estimates")

    pts = pd.read_parquet(interim / "points.parquet")
    pts = pts[pts.match_id == match_id].sort_values("point_index")
    fmt = MatchFormat(int(m.best_of), m.final_set)
    tr = trace(pts, pa, pb, fmt)
    ends = tr[(tr.sets_a + tr.sets_b).diff().fillna(0) > 0]

    fig, ax = plt.subplots(figsize=(9, 4))
    plots.probability_trace(
        ax, tr.point.to_numpy(), {"P(A wins match)": tr.match.to_numpy()},
        [(r.point, f"{r.sets_a}-{r.sets_b}") for r in ends.itertuples()],
    )  # fmt: skip
    ax.set_title(f"{args.title_prefix}{m.player_a} v {m.player_b}: Markov P(match win)")
    ax.set_xlabel(f"Point number (pa={pa:.3f}, pb={pb:.3f})")
    ax.set_ylabel(f"P({m.player_a.split()[-1]} wins)")
    ax.set_xlim(0, tr.point.max())
    return plots.save(fig, Path(args.out) / f"match_trace_{match_id}.png")


if __name__ == "__main__":
    print(main())
