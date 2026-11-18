"""Run the backtest through the C++ core (``courtedge._core``).

Returns the same three DataFrames as :func:`courtedge.market.backtest.run_backtest`, so
callers can switch engines with a flag. ``tests/test_core_parity.py`` checks both
produce identical bets. Only the multiplicative de-vig is implemented in C++.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from courtedge.market.backtest import BacktestConfig, MatchInput

try:
    from courtedge import _core
except ImportError:  # pragma: no cover - exercised when the extension isn't built
    _core = None

STATE_COLS = ["sets_a", "sets_b", "games_a", "games_b", "points_a", "points_b", "server",
              "tiebreak", "tb_first_server", "match_winner"]  # fmt: skip


def available() -> bool:
    return _core is not None


def _ns(times: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def to_core(m: MatchInput, cfg: BacktestConfig):
    fmt = _core.MatchFormat(m.fmt.best_of, m.fmt.final_set.value)
    return _core.MatchData(
        tick_ns=_ns(m.market.time_utc),
        odds_a=m.market.odds_a.to_numpy(float),
        odds_b=m.market.odds_b.to_numpy(float),
        in_play=m.market.in_play.to_numpy(bool),
        point_ns=_ns(m.timeline.time_utc),
        states=m.timeline[STATE_COLS].astype(int).to_numpy(np.int32),
        pa=m.pa, pb=m.pb, fmt=fmt, first_server=m.first_server, winner=m.winner,
        commission=m.commission(cfg),
    )  # fmt: skip


def core_config(cfg: BacktestConfig):
    if cfg.devig_method != "multiplicative":
        raise ValueError("the C++ backtest only implements multiplicative de-vigging")
    s = cfg.staking
    return _core.BacktestConfig(
        delay_s=cfg.delay_s, threshold=cfg.threshold, slippage_ticks=cfg.slippage_ticks,
        min_gap_s=cfg.min_gap_s, max_bets=cfg.max_bets, min_odds=cfg.min_odds,
        max_odds=cfg.max_odds, trade_preplay=cfg.trade_preplay, rule=s.rule, stake=s.stake,
        bankroll=s.bankroll, kelly_fraction=s.kelly_fraction, kelly_cap=s.kelly_cap,
    )  # fmt: skip


def run_backtest_fast(matches: list[MatchInput], cfg: BacktestConfig, threads: int = 0):
    if _core is None:
        raise RuntimeError("courtedge._core is not built; see cpp/README.md")
    out = _core.run_many([to_core(m, cfg) for m in matches], core_config(cfg), threads)
    preds, bets, results = [], [], []
    for m, r in zip(matches, out, strict=True):
        t = pd.to_datetime(r["time_ns"], utc=True)
        mk = m.market.set_index("time_utc").reindex(t)
        preds.append(pd.DataFrame({
            "match_id": m.match_id, "time_utc": t, "odds_a": mk.odds_a.to_numpy(),
            "odds_b": mk.odds_b.to_numpy(), "market_p": r["market_p"],
            "model_game": r["model_game"], "model_set": r["model_set"],
            "model_match": r["model_match"], "known_points": r["known_points"],
            "outcome": int(m.winner == 0),
        }))  # fmt: skip
        for tn, side, odds, stake, mp, kp in r["bets"]:
            t_bet = pd.Timestamp(tn, tz="UTC")
            bets.append(dict(match_id=m.match_id, time_utc=t_bet, side=side, odds=odds,
                             stake=stake, model_p=mp, market_p=kp))  # fmt: skip
        results.append({**r["settlement"], "match_id": m.match_id})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(bets), pd.DataFrame(results)
