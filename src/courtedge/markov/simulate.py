"""Monte Carlo point-by-point simulator, used to check the exact engine.

Same interface as :func:`courtedge.markov.engine.win_probabilities`. It is written
independently of both the engine's recursions and :mod:`courtedge.markov.score`.
It is vectorised over simulations with NumPy and steps every unfinished match
forward one point at a time, so a bug would have to appear in two different
implementations to go unnoticed.

100k simulations from 0-0 in a best-of-5 match take about 2-3 s.
"""

from __future__ import annotations

import numpy as np

from courtedge.markov.engine import WinProbabilities
from courtedge.markov.score import FinalSet, MatchFormat, ScoreState

_NO_TIEBREAK = 10**6


def simulate_win_probabilities(
    state: ScoreState,
    pa: float,
    pb: float,
    fmt: MatchFormat,
    n: int = 100_000,
    seed: int | None = 0,
) -> WinProbabilities:
    """Estimate P(A wins) the current game, set and match by simulation."""
    if state.is_over:
        v = 1.0 if state.winner == 0 else 0.0
        return WinProbabilities(v, v, v)

    rng = np.random.default_rng(seed)
    pts = np.tile(np.array(state.points, dtype=np.int64), (n, 1))
    games = np.tile(np.array(state.games, dtype=np.int64), (n, 1))
    sets = np.tile(np.array(state.sets, dtype=np.int64), (n, 1))
    srv = np.full(n, state.server, dtype=np.int64)
    tb = np.full(n, state.tiebreak)
    tb_first = np.full(n, state.tiebreak_first_server, dtype=np.int64)

    game_winner = np.full(n, -1)
    set_winner = np.full(n, -1)
    match_winner = np.full(n, -1)
    need = fmt.sets_to_win
    final_tb_at = {
        FinalSet.TIEBREAK: 6,
        FinalSet.SUPER_TIEBREAK: 6,
        FinalSet.TIEBREAK_12: 12,
        FinalSet.ADVANTAGE: _NO_TIEBREAK,
    }[fmt.final_set]
    final_tb_target = 10 if fmt.final_set is FinalSet.SUPER_TIEBREAK else 7

    active = np.ones(n, dtype=bool)
    while active.any():
        a = np.flatnonzero(active)
        s = srv[a]
        p_a_point = np.where(s == 0, pa, 1.0 - pb)
        w = np.where(rng.random(a.size) < p_a_point, 0, 1)
        l_ = 1 - w
        pts[a, w] += 1

        is_final = (sets[a, 0] == need - 1) & (sets[a, 1] == need - 1)
        tb_at = np.where(is_final, final_tb_at, 6)
        target = np.where(is_final, final_tb_target, 7)
        pw, pl = pts[a, w], pts[a, l_]

        in_tb = tb[a]
        tb_done = in_tb & (pw >= target) & (pw - pl >= 2)
        game_done = ~in_tb & (pw >= 4) & (pw - pl >= 2)

        # Serve rotation inside a tiebreak that continues.
        rot = in_tb & ~tb_done & ((pts[a, 0] + pts[a, 1]) % 2 == 1)
        srv[a[rot]] = 1 - srv[a[rot]]

        # Regular game finished.
        g = a[game_done]
        wg = w[game_done]
        games[g, wg] += 1
        pts[g] = 0
        srv[g] = 1 - srv[g]
        gw, gl = games[g, wg], games[g, 1 - wg]
        set_by_games = (gw >= 6) & (gw - gl >= 2)
        start_tb = ~set_by_games & (games[g, 0] == games[g, 1]) & (games[g, 0] == tb_at[game_done])
        tb[g[start_tb]] = True
        tb_first[g[start_tb]] = srv[g[start_tb]]

        # Tiebreak finished: it counts as a game and ends the set.
        t = a[tb_done]
        wt = w[tb_done]
        games[t, wt] += 1
        srv[t] = 1 - tb_first[t]

        # Record the first game result after the starting state.
        ended_game = np.zeros(n, dtype=bool)
        ended_game[g] = True
        ended_game[t] = True
        first = ended_game & (game_winner < 0)
        winners_full = np.full(n, -1)
        winners_full[a] = w
        game_winner[first] = winners_full[first]

        # Sets.
        set_done_idx = np.concatenate([g[set_by_games], t])
        set_done_w = np.concatenate([wg[set_by_games], wt])
        sets[set_done_idx, set_done_w] += 1
        games[set_done_idx] = 0
        pts[set_done_idx] = 0
        tb[set_done_idx] = False
        unset = set_winner[set_done_idx] < 0
        set_winner[set_done_idx[unset]] = set_done_w[unset]

        over = sets[set_done_idx, set_done_w] == need
        match_winner[set_done_idx[over]] = set_done_w[over]
        active[set_done_idx[over]] = False

    return WinProbabilities(
        game=float(np.mean(game_winner == 0)),
        set=float(np.mean(set_winner == 0)),
        match=float(np.mean(match_winner == 0)),
    )
