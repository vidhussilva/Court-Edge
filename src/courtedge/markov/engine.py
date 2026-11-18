"""Exact Markov win-probability engine.

Model: points are independent and identically distributed given the server.

    pa = P(A wins a point on A's serve)
    pb = P(B wins a point on B's serve)

The engine nests four levels, and each level only needs the outcome distribution
of the level below:

    point -> game / tiebreak -> set -> match

Loops in the score (deuce, "deuce" in a tiebreak, advantage final sets) are closed
in algebra instead of being recursed forever. Every function is memoised on its
arguments, so a full match trace (around 250 states) costs well under a
millisecond per state after the first call.

All probabilities are for player A.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from courtedge.markov.score import MatchFormat, ScoreState, SetRule

_EPS = 1e-12
# Bounded memo tables: a backtest visits many (pa, pb) pairs, so an unbounded cache
# would keep growing for the whole run.
_CACHE_SIZE = 1 << 20


def _clamp(p: float) -> float:
    # p = 0 or 1 makes some closed forms 0/0; the limit is reached well before 1e-12.
    return min(max(float(p), _EPS), 1.0 - _EPS)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------


def p_game(p: float) -> float:
    """P(server holds) from 0-0, closed form:

        p^4 (15 - 4p - 10p^2 / (1 - 2p(1-p)))

    Comes from summing the ways to win 4-0, 4-1 and 4-2 (p^4 (1 + 4q + 10q^2)) plus
    reaching deuce (20 p^3 q^3) times P(win from deuce) = p^2 / (p^2 + q^2).
    """
    p = _clamp(p)
    return p**4 * (15 - 4 * p - 10 * p**2 / (1 - 2 * p * (1 - p)))


@lru_cache(maxsize=_CACHE_SIZE)
def p_game_from(p: float, a: int, b: int) -> float:
    """P(server wins the game) when the server has ``a`` points and the returner ``b``.

    Uses the normalised score from :mod:`courtedge.markov.score` (deuce = (3, 3)).
    """
    p = _clamp(p)
    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0
    if a >= 3 and b >= 3:
        # Deuce: win two in a row before losing two in a row.
        deuce = p * p / (p * p + (1 - p) * (1 - p))
        if a == b:
            return deuce
        return p + (1 - p) * deuce if a > b else p * deuce
    return p * p_game_from(p, a + 1, b) + (1 - p) * p_game_from(p, a, b + 1)


def _p_point_a(server: int, pa: float, pb: float) -> float:
    return pa if server == 0 else 1.0 - pb


def p_game_a(pa: float, pb: float, points: tuple[int, int], server: int) -> float:
    """P(A wins the current regular game) from a score (points_a, points_b)."""
    if server == 0:
        return p_game_from(_clamp(pa), points[0], points[1])
    return 1.0 - p_game_from(_clamp(pb), points[1], points[0])


# ---------------------------------------------------------------------------
# Tiebreak
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_CACHE_SIZE)
def p_tiebreak(pa: float, pb: float, a: int, b: int, server: int, target: int = 7) -> float:
    """P(A wins a tiebreak to ``target`` (win by two) from (a, b) with ``server`` next.

    The serve changes after every odd total of points played (A, B B, A A, ...).
    """
    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0
    if a >= target - 1 and b >= target - 1 and a == b:
        # Level at target-1 or beyond. An even number of points has been played, so
        # the next two points are served by `server` then the other player, and the
        # serve then stays with the other player. Solve the two equations:
        #   x_s = w_s + t_s * x_o,   x_o = w_o + t_o * x_s
        # where w = P(A wins both points) and t = P(the pair is split).
        def pair(s: int) -> tuple[float, float]:
            q1, q2 = _p_point_a(s, pa, pb), _p_point_a(1 - s, pa, pb)
            return q1 * q2, q1 * (1 - q2) + (1 - q1) * q2

        w_s, t_s = pair(server)
        w_o, t_o = pair(1 - server)
        return (w_s + t_s * w_o) / (1 - t_s * t_o)
    if a >= target - 1 and b >= target - 1:
        # Advantage: re-normalise to the level state after one more point.
        lead = 0 if a > b else 1
        q = _p_point_a(server, pa, pb)
        nxt = 1 - server if (a + b + 1) % 2 == 1 else server
        level = p_tiebreak(pa, pb, target - 1, target - 1, nxt, target)
        return q * (1.0 if lead == 0 else level) + (1 - q) * (level if lead == 0 else 0.0)
    q = _p_point_a(server, pa, pb)
    nxt = 1 - server if (a + b + 1) % 2 == 1 else server
    return q * p_tiebreak(pa, pb, a + 1, b, nxt, target) + (1 - q) * p_tiebreak(
        pa, pb, a, b + 1, nxt, target
    )


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

# Outcome of a set: probabilities indexed by (winner, server of the next set's first game).
SetOutcome = tuple[float, float, float, float]  # (A,0), (A,1), (B,0), (B,1)


def _outcome(winner: int, next_server: int, prob: float) -> SetOutcome:
    out = [0.0, 0.0, 0.0, 0.0]
    out[winner * 2 + next_server] = prob
    return tuple(out)  # type: ignore[return-value]


def _add(x: SetOutcome, y: SetOutcome, wx: float = 1.0, wy: float = 1.0) -> SetOutcome:
    return tuple(wx * a + wy * b for a, b in zip(x, y, strict=True))  # type: ignore[return-value]


@lru_cache(maxsize=_CACHE_SIZE)
def set_outcomes(
    pa: float, pb: float, ga: int, gb: int, server: int, rule: SetRule
) -> SetOutcome:
    """Distribution of (set winner, next server) from the start of a game at (ga, gb)."""
    if rule.tiebreak_at is not None and ga == gb == rule.tiebreak_at:
        p = p_tiebreak(pa, pb, 0, 0, server, rule.tiebreak_points)
        # The tiebreak's first receiver serves first in the next set.
        return _add(_outcome(0, 1 - server, p), _outcome(1, 1 - server, 1 - p))

    g = p_game_a(pa, pb, (0, 0), server)  # P(A wins this game)
    if rule.tiebreak_at is None and ga == gb and ga >= 6:
        # Advantage set, level: need two games in a row. The pair is served by
        # `server` then the other player, so the set always ends with the other
        # player serving and `server` serves first in the next set.
        g2 = p_game_a(pa, pb, (0, 0), 1 - server)
        both_a, both_b = g * g2, (1 - g) * (1 - g2)
        p = both_a / (both_a + both_b)
        return _add(_outcome(0, server, p), _outcome(1, server, 1 - p))

    nxt = 1 - server
    total: SetOutcome = (0.0, 0.0, 0.0, 0.0)
    for winner, prob in ((0, g), (1, 1 - g)):
        na, nb = (ga + 1, gb) if winner == 0 else (ga, gb + 1)
        hi, lo = max(na, nb), min(na, nb)
        if hi >= 6 and hi - lo >= 2:
            branch = _outcome(winner, nxt, 1.0)
        else:
            branch = set_outcomes(pa, pb, na, nb, nxt, rule)
        total = _add(total, branch, 1.0, prob)
    return total


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_CACHE_SIZE)
def p_match_from_set_start(
    pa: float, pb: float, sa: int, sb: int, server: int, fmt: MatchFormat
) -> float:
    """P(A wins the match) at the start of a set with set score (sa, sb)."""
    need = fmt.sets_to_win
    if sa == need:
        return 1.0
    if sb == need:
        return 0.0
    out = set_outcomes(pa, pb, 0, 0, server, fmt.set_rule(sa, sb))
    return (
        out[0] * p_match_from_set_start(pa, pb, sa + 1, sb, 0, fmt)
        + out[1] * p_match_from_set_start(pa, pb, sa + 1, sb, 1, fmt)
        + out[2] * p_match_from_set_start(pa, pb, sa, sb + 1, 0, fmt)
        + out[3] * p_match_from_set_start(pa, pb, sa, sb + 1, 1, fmt)
    )


@dataclass(frozen=True)
class WinProbabilities:
    game: float  # P(A wins the current game; a tiebreak counts as the game)
    set: float  # P(A wins the current set)
    match: float  # P(A wins the match)


def _current_set_outcomes(state: ScoreState, pa: float, pb: float, fmt: MatchFormat):
    """(P(A wins current game), set outcome distribution) from a mid-game state."""
    rule = fmt.set_rule(*state.sets)
    if state.tiebreak:
        p = p_tiebreak(pa, pb, state.points[0], state.points[1], state.server,
                       rule.tiebreak_points)  # fmt: skip
        nxt = 1 - state.tiebreak_first_server
        return p, _add(_outcome(0, nxt, p), _outcome(1, nxt, 1 - p))

    g = p_game_a(pa, pb, state.points, state.server)
    nxt = 1 - state.server
    total: SetOutcome = (0.0, 0.0, 0.0, 0.0)
    for winner, prob in ((0, g), (1, 1 - g)):
        ga, gb = state.games
        na, nb = (ga + 1, gb) if winner == 0 else (ga, gb + 1)
        hi, lo = max(na, nb), min(na, nb)
        if hi >= 6 and hi - lo >= 2:
            branch = _outcome(winner, nxt, 1.0)
        else:
            branch = set_outcomes(pa, pb, na, nb, nxt, rule)
        total = _add(total, branch, 1.0, prob)
    return g, total


def win_probabilities(
    state: ScoreState, pa: float, pb: float, fmt: MatchFormat
) -> WinProbabilities:
    """P(A wins) the current game, set and match from any score state."""
    if state.is_over:
        v = 1.0 if state.winner == 0 else 0.0
        return WinProbabilities(v, v, v)
    pa, pb = _clamp(pa), _clamp(pb)
    g, out = _current_set_outcomes(state, pa, pb, fmt)
    sa, sb = state.sets
    match = (
        out[0] * p_match_from_set_start(pa, pb, sa + 1, sb, 0, fmt)
        + out[1] * p_match_from_set_start(pa, pb, sa + 1, sb, 1, fmt)
        + out[2] * p_match_from_set_start(pa, pb, sa, sb + 1, 0, fmt)
        + out[3] * p_match_from_set_start(pa, pb, sa, sb + 1, 1, fmt)
    )
    return WinProbabilities(game=g, set=out[0] + out[1], match=match)


def p_match(pa: float, pb: float, fmt: MatchFormat, first_server: int = 0) -> float:
    """Pre-match P(A wins)."""
    return win_probabilities(ScoreState(server=first_server), pa, pb, fmt).match


def clear_caches() -> None:
    for f in (p_game_from, p_tiebreak, set_outcomes, p_match_from_set_start):
        f.cache_clear()
