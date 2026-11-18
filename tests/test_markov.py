"""Markov engine tests.

1. Closed forms and identities (exact).
2. Martingale check: P(state) = q * P(state after A wins point) + (1-q) * P(after B wins),
   using the separate score state machine to produce the next states. That ties the
   engine to the score rules at every state of a real-format match, to 1e-12.
3. Monte Carlo grid from the spec: 100k simulations, tolerance 0.005.
"""

import zlib
from pathlib import Path

import numpy as np
import pytest

from courtedge.data import pbp
from courtedge.markov.engine import (
    p_game,
    p_game_from,
    p_match,
    p_tiebreak,
    win_probabilities,
)
from courtedge.markov.score import FinalSet, MatchFormat, ScoreState, advance
from courtedge.markov.simulate import simulate_win_probabilities

FIX = Path(__file__).parent / "fixtures"
ALL_RULES = list(FinalSet)


def test_game_closed_form_value():
    assert p_game(0.6) == pytest.approx(0.736, abs=5e-4)
    assert p_game(0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("p", np.linspace(0.3, 0.9, 13))
def test_game_closed_form_matches_recursion(p):
    assert p_game(p) == pytest.approx(p_game_from(float(p), 0, 0), abs=1e-12)


def test_deuce_and_advantage():
    p = 0.6
    deuce = p * p / (p * p + 0.16)
    assert p_game_from(p, 3, 3) == pytest.approx(deuce)
    assert p_game_from(p, 4, 3) == pytest.approx(p + (1 - p) * deuce)
    assert p_game_from(p, 3, 4) == pytest.approx(p * deuce)


@pytest.mark.parametrize("rule", ALL_RULES)
@pytest.mark.parametrize("best_of", [3, 5])
def test_equal_players_are_coin_flips(rule, best_of):
    fmt = MatchFormat(best_of, rule)
    assert p_match(0.63, 0.63, fmt) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("rule", ALL_RULES)
def test_symmetry_swapping_players(rule):
    fmt = MatchFormat(5, rule)
    assert p_match(0.67, 0.61, fmt, 0) == pytest.approx(1 - p_match(0.61, 0.67, fmt, 1))


def test_longer_formats_favour_the_stronger_player():
    bo3 = p_match(0.66, 0.62, MatchFormat(3))
    bo5 = p_match(0.66, 0.62, MatchFormat(5))
    assert 0.5 < bo3 < bo5


def test_tiebreak_matches_brute_force():
    # Independent simulation that encodes the serve order directly by point number.
    pa, pb = 0.65, 0.6
    exact = p_tiebreak(pa, pb, 0, 0, 0, 7)
    rng = np.random.default_rng(1)
    n = 200_000
    a = np.zeros(n, int)
    b = np.zeros(n, int)
    done = np.zeros(n, bool)
    a_won = np.zeros(n, bool)
    k = 0
    while not done.all():
        server = 0 if ((k + 1) // 2) % 2 == 0 else 1  # A, B, B, A, A, ...
        q = pa if server == 0 else 1 - pb
        win = rng.random(n) < q
        a += win & ~done
        b += ~win & ~done
        fin = ~done & (((a >= 7) & (a - b >= 2)) | ((b >= 7) & (b - a >= 2)))
        a_won[fin] = a[fin] > b[fin]
        done |= fin
        k += 1
    wins = a_won.sum()
    assert exact == pytest.approx(wins / n, abs=0.005)


def _martingale_ok(state, pa, pb, fmt):
    q = pa if state.server == 0 else 1 - pb
    now = win_probabilities(state, pa, pb, fmt)
    after_a = advance(state, 0, fmt)
    after_b = advance(state, 1, fmt)
    wa = win_probabilities(after_a, pa, pb, fmt)
    wb = win_probabilities(after_b, pa, pb, fmt)
    assert now.match == pytest.approx(q * wa.match + (1 - q) * wb.match, abs=1e-12)
    same_set_a = after_a.sets == state.sets and not after_a.is_over
    same_set_b = after_b.sets == state.sets and not after_b.is_over
    set_a = wa.set if same_set_a else float(after_a.sets[0] > state.sets[0])
    set_b = wb.set if same_set_b else float(after_b.sets[0] > state.sets[0])
    assert now.set == pytest.approx(q * set_a + (1 - q) * set_b, abs=1e-12)


def test_martingale_along_fixture_match():
    points, _ = pbp.load_slam(FIX / "slam" / "2023-wimbledon-points.csv")
    g = points[points.match_id == "2023-wimbledon-1101"]
    fmt = MatchFormat(5, FinalSet.SUPER_TIEBREAK)
    for _, row in g.iterrows():
        _martingale_ok(pbp.state_at(row), 0.66, 0.64, fmt)


@pytest.mark.parametrize("rule", ALL_RULES)
def test_martingale_random_states(rule):
    fmt = MatchFormat(5, rule)
    rng = np.random.default_rng(3)
    state = ScoreState(sets=(2, 2), games=(5, 5))
    for _ in range(400):  # long deciding sets reach 12-12 and advantage play
        if state.is_over:
            state = ScoreState(sets=(2, 2), games=(5, 5), server=int(rng.integers(2)))
        _martingale_ok(state, 0.7, 0.68, fmt)
        state = advance(state, int(rng.random() < 0.5), fmt)


# ---------------------------------------------------------------------------
# Monte Carlo grid (spec: 100k sims, p in 0.50..0.75 step 0.05, tolerance 0.005)
# ---------------------------------------------------------------------------

GRID_STATES = {
    "bo3 start": (MatchFormat(3), ScoreState()),
    "bo3 deuce, receiver serving": (
        MatchFormat(3), ScoreState(sets=(1, 0), games=(4, 5), points=(3, 3), server=1)),
    "bo3 tiebreak 5-5": (MatchFormat(3), ScoreState(
        games=(6, 6), points=(5, 5), tiebreak=True, server=0, tiebreak_first_server=1)),
    "bo5 start": (MatchFormat(5, FinalSet.SUPER_TIEBREAK), ScoreState()),
    "bo5 2-2 super TB": (MatchFormat(5, FinalSet.SUPER_TIEBREAK), ScoreState(sets=(2, 2))),
    "bo5 2-2 advantage 6-6": (
        MatchFormat(5, FinalSet.ADVANTAGE), ScoreState(sets=(2, 2), games=(6, 6), server=1)),
    "bo5 2-2 TB at 12": (MatchFormat(5, FinalSet.TIEBREAK_12), ScoreState(sets=(2, 2))),
}  # fmt: skip


@pytest.mark.slow
@pytest.mark.parametrize("name", list(GRID_STATES))
@pytest.mark.parametrize("p", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
def test_engine_matches_monte_carlo(name, p):
    fmt, state = GRID_STATES[name]
    pb = 0.62  # asymmetric on purpose, so a server mix-up would show
    exact = win_probabilities(state, p, pb, fmt)
    mc = simulate_win_probabilities(state, p, pb, fmt, n=100_000, seed=zlib.crc32(name.encode()))
    assert exact.game == pytest.approx(mc.game, abs=0.005)
    assert exact.set == pytest.approx(mc.set, abs=0.005)
    assert exact.match == pytest.approx(mc.match, abs=0.005)


def test_monte_carlo_quick():
    fmt, state = GRID_STATES["bo3 start"]
    exact = win_probabilities(state, 0.64, 0.6, fmt)
    mc = simulate_win_probabilities(state, 0.64, 0.6, fmt, n=20_000, seed=5)
    assert exact.match == pytest.approx(mc.match, abs=0.015)


@pytest.mark.parametrize(
    "state",
    [
        ScoreState(points=(4, 4)),
        ScoreState(games=(7, 7)),
        ScoreState(games=(6, 6)),  # a standard set at 6-6 must be in a tiebreak
        ScoreState(games=(5, 5), tiebreak=True),
        ScoreState(sets=(2, 0)),
    ],
)
def test_invalid_states_are_rejected(state):
    with pytest.raises(ValueError):
        win_probabilities(state, 0.65, 0.62, MatchFormat(3))
