"""Python engine vs the C++ extension (skipped if courtedge._core is not built).

Build it with:
    uv run cmake -S cpp -B build -DCOURTEDGE_BUILD_PYTHON=ON \
        -DPython_EXECUTABLE=$(uv run which python)
    cmake --build build
"""

import numpy as np
import pytest

from courtedge.markov import engine
from courtedge.markov.score import FinalSet, MatchFormat, ScoreState, advance

core = pytest.importorskip("courtedge._core")


def to_core(s: ScoreState):
    return core.ScoreState(
        sets=s.sets, games=s.games, points=s.points, server=s.server, tiebreak=s.tiebreak,
        tiebreak_first_server=s.tiebreak_first_server,
        winner=-1 if s.winner is None else s.winner,
    )  # fmt: skip


@pytest.mark.parametrize("rule", list(FinalSet))
def test_advance_and_engine_agree_on_random_walks(rule):
    fmt = MatchFormat(5, rule)
    cfmt = core.MatchFormat(5, rule.value)
    rng = np.random.default_rng(11)
    for _ in range(5):
        pa, pb = rng.uniform(0.55, 0.75, 2)
        ceng = core.MarkovEngine(pa, pb, cfmt)
        s = ScoreState(sets=(2, 2)) if rng.random() < 0.5 else ScoreState()
        cs = to_core(s)
        while not s.is_over:
            py_w = engine.win_probabilities(s, pa, pb, fmt)
            c_w = ceng.win_probabilities(cs)
            assert c_w.match == pytest.approx(py_w.match, abs=1e-10)
            assert c_w.set == pytest.approx(py_w.set, abs=1e-10)
            assert c_w.game == pytest.approx(py_w.game, abs=1e-10)
            w = int(rng.integers(2))
            s, cs = advance(s, w, fmt), core.advance(cs, w, cfmt)
            assert cs == to_core(s)


def test_simulate_binding_releases_gil_and_agrees():
    fmt = core.MatchFormat(3, "tiebreak")
    mc = core.simulate(core.ScoreState(), 0.65, 0.62, fmt, n=200_000, seed=1, threads=2)
    exact = core.MarkovEngine(0.65, 0.62, fmt).p_match()
    assert mc.match == pytest.approx(exact, abs=0.005)


def test_backtest_parity_python_vs_cpp():
    """Same inputs, same config: identical bets and P&L, predictions to 1e-10."""
    from dataclasses import replace

    from courtedge.market import fast
    from courtedge.market.backtest import BacktestConfig, run_backtest
    from courtedge.market.staking import StakingConfig
    from tests.test_backtest import _fixture_input

    m = _fixture_input()
    for cfg in [
        BacktestConfig(delay_s=120, threshold=0.02, min_gap_s=0, max_bets=1000),
        BacktestConfig(
            delay_s=0,
            threshold=0.03,
            slippage_ticks=2,
            staking=StakingConfig("kelly", kelly_cap=0.5),
        ),  # fmt: skip
        BacktestConfig(delay_s=300, threshold=0.01, min_gap_s=600, trade_preplay=True),
    ]:
        inputs = [m, replace(m, match_id="copy", pa=0.62, pb=0.7, market_base_rate=2.0)]
        p_py, b_py, r_py = run_backtest(inputs, cfg)
        p_c, b_c, r_c = fast.run_backtest_fast(inputs, cfg, threads=2)
        assert len(b_py) > 0
        np.testing.assert_allclose(p_c.model_match, p_py.model_match, atol=1e-10)
        np.testing.assert_allclose(p_c.market_p, p_py.market_p, atol=1e-12)
        assert (p_c.known_points.to_numpy() == p_py.known_points.to_numpy()).all()
        assert list(b_c.side) == list(b_py.side)
        assert list(b_c.time_utc) == list(b_py.time_utc)
        np.testing.assert_allclose(b_c.odds, b_py.odds, atol=1e-9)
        np.testing.assert_allclose(b_c.stake, b_py.stake, atol=1e-12)
        np.testing.assert_allclose(r_c.net, r_py.net, atol=1e-9)
