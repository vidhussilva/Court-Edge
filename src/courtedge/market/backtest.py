"""Event-driven backtester for one market at a time.

At each market tick t (one per minute in BASIC data):

1. Market probability: de-vig the last traded prices at t.
2. Model probability: Markov engine on the **last score state known at t - delay**
   (broadcast delay) and the pre-match serve probabilities.
3. If model - market > threshold for either player, back that player at the price at t,
   made worse by ``slippage_ticks`` ticks (BASIC data has no order book; see the report).
4. When the market closes, settle every bet with :func:`settle_market`.

No look-ahead. :func:`run_match` gets the whole match, but each decision only reads
rows with ``time <= t`` (market) and ``time <= t - delay`` (score). Both lookups are
in :class:`_AsOf`. ``tests/test_backtest.py`` checks this by perturbing everything
after a cut-off and asserting that earlier decisions don't change.

Simulation only: nothing here places bets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from courtedge.market.devig import devig
from courtedge.market.staking import StakingConfig, stake_for
from courtedge.market.ticks import tick_down
from courtedge.markov.engine import win_probabilities
from courtedge.markov.score import MatchFormat, ScoreState

HORIZONS = ("game", "set", "match")
DEFAULT_COMMISSION = 0.05  # most commonly quoted UK Market Base Rate; see reports/phase2.md


@dataclass(frozen=True)
class BacktestConfig:
    delay_s: float = 60.0
    threshold: float = 0.05
    # Commission on net market winnings. None = use each market's recorded
    # marketBaseRate from the Betfair file, falling back to DEFAULT_COMMISSION.
    commission: float | None = None
    slippage_ticks: int = 1
    devig_method: str = "multiplicative"
    min_gap_s: float = 300.0  # at most one bet per 5 minutes per market
    max_bets: int = 10
    min_odds: float = 1.05  # ignore prices where one tick of slippage is most of the edge
    max_odds: float = 20.0
    trade_preplay: bool = False
    staking: StakingConfig = field(default_factory=StakingConfig)

    @classmethod
    def from_dict(cls, d: dict) -> BacktestConfig:
        d = dict(d)
        staking = StakingConfig(**d.pop("staking", {}))
        return cls(staking=staking, **d)


@dataclass(frozen=True)
class MatchInput:
    """Everything the backtester needs for one match (times in UTC)."""

    match_id: str
    market: pd.DataFrame  # time_utc, odds_a, odds_b, in_play; OPEN ticks only, sorted
    timeline: pd.DataFrame  # score_timeline(): time_utc + state after each point, sorted
    pa: float
    pb: float
    fmt: MatchFormat
    first_server: int
    winner: int  # 0 = A, 1 = B; used only by settle_market
    market_base_rate: float | None = None  # commission in percent, from the Betfair file

    def commission(self, cfg: BacktestConfig) -> float:
        if cfg.commission is not None:
            return cfg.commission
        if self.market_base_rate is not None and self.market_base_rate == self.market_base_rate:
            return self.market_base_rate / 100.0
        return DEFAULT_COMMISSION


@dataclass(frozen=True)
class Bet:
    match_id: str
    time_utc: pd.Timestamp
    side: int  # 0 = backed A, 1 = backed B
    odds: float  # price after slippage
    stake: float
    model_p: float  # model P(side wins)
    market_p: float  # de-vigged market P(side wins)


class _AsOf:
    """The only window through which decisions see the data."""

    def __init__(self, m: MatchInput):
        self.tl_times = m.timeline["time_utc"].to_numpy(dtype="datetime64[ns]")
        self.tl = m.timeline
        self.first = ScoreState(server=m.first_server)

    def state(self, t: np.datetime64) -> tuple[ScoreState, int]:
        """Last known score at time t and how many points were known by then."""
        i = int(np.searchsorted(self.tl_times, t, side="right")) - 1
        if i < 0:
            return self.first, 0
        r = self.tl.iloc[i]
        winner = None if r["match_winner"] < 0 else int(r["match_winner"])
        state = ScoreState(
            sets=(int(r.sets_a), int(r.sets_b)), games=(int(r.games_a), int(r.games_b)),
            points=(int(r.points_a), int(r.points_b)), server=int(r.server),
            tiebreak=bool(r.tiebreak), tiebreak_first_server=int(r.tb_first_server),
            winner=winner,
        )  # fmt: skip
        return state, i + 1


def predictions(m: MatchInput, cfg: BacktestConfig) -> pd.DataFrame:
    """Model and market P(A wins) at every tick, the model lagged by ``cfg.delay_s``."""
    view = _AsOf(m)
    delay = np.timedelta64(int(cfg.delay_s * 1e9), "ns")
    mk = m.market if cfg.trade_preplay else m.market[m.market["in_play"]]
    rows = []
    for r in mk.itertuples(index=False):
        t = np.datetime64(r.time_utc.tz_convert(None) if r.time_utc.tzinfo else r.time_utc, "ns")
        state, n_known = view.state(t - delay)
        w = win_probabilities(state, m.pa, m.pb, m.fmt)
        rows.append(
            dict(match_id=m.match_id, time_utc=r.time_utc, odds_a=r.odds_a, odds_b=r.odds_b,
                 market_p=float(devig(r.odds_a, r.odds_b, cfg.devig_method)),
                 model_game=w.game, model_set=w.set, model_match=w.match,
                 known_points=n_known)
        )  # fmt: skip
    return pd.DataFrame(rows)


def decide(preds: pd.DataFrame, m: MatchInput, cfg: BacktestConfig) -> list[Bet]:
    """Turn the prediction log into bets. Each row only uses its own columns."""
    bets: list[Bet] = []
    last_t = None
    for r in preds.itertuples(index=False):
        if len(bets) >= cfg.max_bets:
            break
        if last_t is not None and (r.time_utc - last_t).total_seconds() < cfg.min_gap_s:
            continue
        edge_a = r.model_match - r.market_p
        if abs(edge_a) <= cfg.threshold:
            continue
        side = 0 if edge_a > 0 else 1
        raw_odds = r.odds_a if side == 0 else r.odds_b
        if not (cfg.min_odds <= raw_odds <= cfg.max_odds):
            continue
        odds = tick_down(raw_odds, cfg.slippage_ticks)
        model_p = r.model_match if side == 0 else 1 - r.model_match
        market_p = r.market_p if side == 0 else 1 - r.market_p
        stake = stake_for(model_p, odds, cfg.staking, m.commission(cfg))
        if stake <= 0:
            continue
        bets.append(Bet(m.match_id, r.time_utc, side, odds, stake, model_p, market_p))
        last_t = r.time_utc
    return bets


def bet_pnl(bet: Bet, winner: int) -> float:
    """Gross profit of one back bet before commission."""
    return bet.stake * (bet.odds - 1.0) if bet.side == winner else -bet.stake


def settle_market(bets: list[Bet], winner: int, commission: float) -> dict:
    """Settle all bets in one market the way Betfair does.

    Commission is charged on *net* market winnings: losing bets in the same market
    offset winning ones, and a net loss pays no commission.
    """
    gross = float(sum(bet_pnl(b, winner) for b in bets))
    comm = commission * gross if gross > 0 else 0.0
    staked = float(sum(b.stake for b in bets))
    return dict(n_bets=len(bets), staked=staked, gross=gross, commission=comm, net=gross - comm)


def run_match(m: MatchInput, cfg: BacktestConfig) -> tuple[pd.DataFrame, list[Bet], dict]:
    preds = predictions(m, cfg)
    bets = decide(preds, m, cfg) if len(preds) else []
    result = settle_market(bets, m.winner, m.commission(cfg))
    result["match_id"] = m.match_id
    preds["outcome"] = int(m.winner == 0)
    return preds, bets, result


def run_backtest(matches: list[MatchInput], cfg: BacktestConfig):
    """Run every match. Returns (predictions, bets, per-match results) as DataFrames."""
    all_preds, all_bets, results = [], [], []
    for m in matches:
        p, b, r = run_match(m, cfg)
        all_preds.append(p)
        all_bets.extend(asdict(x) for x in b)
        results.append(r)
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return preds, pd.DataFrame(all_bets), pd.DataFrame(results)
