from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtedge.data import betfair, pbp
from courtedge.data.catalogue import build_catalogue, market_summary
from courtedge.market.backtest import (
    BacktestConfig,
    Bet,
    MatchInput,
    predictions,
    run_match,
    settle_market,
)
from courtedge.market.inputs import build_inputs
from courtedge.market.staking import StakingConfig, kelly_fraction, stake_for
from courtedge.market.ticks import tick_down, to_tick

FIX = Path(__file__).parent / "fixtures"
T = pd.Timestamp("2023-07-16 15:00", tz="UTC")


def _fixture_input() -> MatchInput:
    points, matches = pbp.load_slam(FIX / "slam" / "2023-wimbledon-points.csv")
    bf = betfair.parse_path(FIX / "betfair")
    cat, _ = build_catalogue(matches, market_summary(bf))
    serve = pd.DataFrame({"match_id": cat.match_id, "pa": [0.68], "pb": [0.64]})
    inputs, info = build_inputs(cat, matches, points, bf, serve)
    assert info.used.all()
    return inputs[0]


@pytest.fixture(scope="module")
def match_input() -> MatchInput:
    return _fixture_input()


# A low threshold and no spacing, so the fixture produces plenty of bets.
ACTIVE = BacktestConfig(delay_s=120, threshold=0.02, min_gap_s=0, max_bets=1000)


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------


def _perturb_after(m: MatchInput, cut: pd.Timestamp, rng) -> MatchInput:
    mk = m.market.copy()
    late = mk.time_utc > cut
    mk.loc[late, "odds_a"] = rng.uniform(1.1, 15, late.sum()).round(2)
    mk.loc[late, "odds_b"] = rng.uniform(1.1, 15, late.sum()).round(2)
    tl = m.timeline.copy()
    late_tl = tl.time_utc > cut
    # Shuffle whole late score states (each stays a valid tennis score).
    state_cols = ["sets_a", "sets_b", "games_a", "games_b", "points_a", "points_b", "server",
                  "tiebreak", "tb_first_server", "match_winner"]  # fmt: skip
    late_rows = tl.loc[late_tl, state_cols].to_numpy()
    tl.loc[late_tl, state_cols] = late_rows[rng.permutation(len(late_rows))]
    # Even the result changes; decisions before the cut must not care.
    return replace(m, market=mk, timeline=tl, winner=1 - m.winner)


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.8])
def test_no_lookahead(match_input, frac):
    m = match_input
    in_play = m.market[m.market.in_play].time_utc
    cut = in_play.iloc[int(len(in_play) * frac)]
    preds0, bets0, _ = run_match(m, ACTIVE)
    preds1, bets1, _ = run_match(_perturb_after(m, cut, np.random.default_rng(0)), ACTIVE)
    assert len(bets0) > 5, "fixture should trigger bets, otherwise the test proves nothing"

    early0 = preds0[preds0.time_utc <= cut].drop(columns="outcome")
    early1 = preds1[preds1.time_utc <= cut].drop(columns="outcome")
    pd.testing.assert_frame_equal(early0.reset_index(drop=True), early1.reset_index(drop=True))
    assert [b for b in bets0 if b.time_utc <= cut] == [b for b in bets1 if b.time_utc <= cut]
    # And the perturbation really did change things after the cut.
    assert not preds0[preds0.time_utc > cut].model_match.equals(
        preds1[preds1.time_utc > cut].model_match.reset_index(drop=True)
    )


def test_delay_hides_recent_points(match_input):
    m = match_input
    p0 = predictions(m, replace(ACTIVE, delay_s=0))
    p5 = predictions(m, replace(ACTIVE, delay_s=300))
    tl = m.timeline.time_utc
    for i in [10, 50, 120]:
        t = p0.time_utc.iloc[i]
        assert p0.known_points.iloc[i] == (tl <= t).sum()
        assert p5.known_points.iloc[i] == (tl <= t - pd.Timedelta(seconds=300)).sum()
    assert (p5.known_points <= p0.known_points).all()


# ---------------------------------------------------------------------------
# P&L accounting (hand-calculated)
# ---------------------------------------------------------------------------


def bet(side, odds, stake):
    return Bet("m", T, side, odds, stake, 0.5, 0.5)


def test_single_winning_back_bet():
    r = settle_market([bet(0, 2.5, 10)], winner=0, commission=0.05)
    assert r["gross"] == pytest.approx(15.0)
    assert r["commission"] == pytest.approx(0.75)
    assert r["net"] == pytest.approx(14.25)


def test_losing_bet_pays_no_commission():
    r = settle_market([bet(0, 2.5, 10)], winner=1, commission=0.05)
    assert r["net"] == pytest.approx(-10.0) and r["commission"] == 0.0


def test_commission_on_net_market_winnings():
    bets = [bet(0, 2.0, 10), bet(1, 3.0, 5)]
    a = settle_market(bets, winner=0, commission=0.05)  # +10 - 5 = +5
    assert (a["gross"], a["commission"], a["net"]) == pytest.approx((5.0, 0.25, 4.75))
    b = settle_market(bets, winner=1, commission=0.05)  # -10 + 10 = 0
    assert (b["gross"], b["commission"], b["net"]) == pytest.approx((0.0, 0.0, 0.0))
    assert a["staked"] == 15 and a["n_bets"] == 2


def test_no_bets_settles_to_zero():
    assert settle_market([], 0, 0.05)["net"] == 0.0


def test_market_base_rate_used_when_config_is_none(match_input):
    m = replace(match_input, market_base_rate=2.0)
    assert m.commission(BacktestConfig(commission=None)) == pytest.approx(0.02)
    assert m.commission(BacktestConfig(commission=0.07)) == pytest.approx(0.07)
    assert replace(m, market_base_rate=None).commission(BacktestConfig()) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Staking and ticks
# ---------------------------------------------------------------------------


def test_kelly_hand_values():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly_fraction(0.6, 2.0, 0.05) == pytest.approx((0.6 * 0.95 - 0.4) / 0.95)
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_stake_rules():
    assert stake_for(0.6, 2.0, StakingConfig("flat", stake=2.0)) == 2.0
    kelly = StakingConfig("kelly", bankroll=100, kelly_fraction=0.25, kelly_cap=0.5)
    assert stake_for(0.6, 2.0, kelly) == pytest.approx(5.0)  # 100 * 0.25 * 0.2
    capped = StakingConfig("kelly", bankroll=100, kelly_fraction=1.0, kelly_cap=0.02)
    assert stake_for(0.6, 2.0, capped) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        stake_for(0.6, 2.0, StakingConfig("martingale"))


@pytest.mark.parametrize(
    "price, down",
    [
        (1.5, 1.49),
        (2.0, 1.99),
        (2.02, 2.0),
        (3.0, 2.98),
        (4.1, 4.0),
        (10.0, 9.8),
        (10.5, 10.0),
        (1000, 990),
        (1.01, 1.01),
    ],
)
def test_tick_down(price, down):
    assert tick_down(price) == down


def test_to_tick_rounds_onto_ladder():
    assert to_tick(2.013) == 2.02 or to_tick(2.013) == 2.0
    assert to_tick(4.07) == 4.1 and to_tick(0.5) == 1.01 and to_tick(5000) == 1000


def test_config_from_dict():
    cfg = BacktestConfig.from_dict({"delay_s": 120, "staking": {"rule": "kelly"}})
    assert cfg.delay_s == 120 and cfg.staking.rule == "kelly"
