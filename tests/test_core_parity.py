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
