"""Tennis score state machine.

Everything else (the parsers, the Markov engine, the Monte Carlo simulator and the
backtester) uses this one definition of the score, so they all agree on things like
who serves after a tiebreak.

Conventions
-----------
* Players are indexed 0 (player A / "P1") and 1 (player B / "P2").
* ``server`` is the player who serves the *next* point.
* In a regular game ``points`` counts points won (0, 1, 2, 3 = 0/15/30/40).
  Deuce-like scores are normalised to (3, 3) and advantages to (4, 3) / (3, 4), so
  every reachable state has a canonical form. That keeps memo tables small.
* In a tiebreak ``points`` counts tiebreak points and is normalised the same way
  once both players reach target - 1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class FinalSet(StrEnum):
    """How the deciding set is played."""

    TIEBREAK = "tiebreak"  # 7-point tiebreak at 6-6 (pre-2019 US Open, ATP tour)
    ADVANTAGE = "advantage"  # no tiebreak; win by two games (old Wimbledon/RG/AO)
    TIEBREAK_12 = "tiebreak_12"  # 7-point tiebreak at 12-12 (Wimbledon 2019-2021)
    SUPER_TIEBREAK = "super_tiebreak"  # 10-point tiebreak at 6-6 (all slams 2022+)


@dataclass(frozen=True)
class SetRule:
    tiebreak_at: int | None  # games each at which a tiebreak starts; None = advantage set
    tiebreak_points: int  # points needed to win the tiebreak (win by two)


@dataclass(frozen=True)
class MatchFormat:
    best_of: int = 3
    final_set: FinalSet = FinalSet.TIEBREAK

    def __post_init__(self) -> None:
        if self.best_of not in (1, 3, 5):
            raise ValueError(f"best_of must be 1, 3 or 5, got {self.best_of}")
        object.__setattr__(self, "final_set", FinalSet(self.final_set))

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1

    def set_rule(self, sets_a: int, sets_b: int) -> SetRule:
        """Rule for the set about to be played when the set score is (sets_a, sets_b)."""
        is_final = sets_a == sets_b == self.sets_to_win - 1
        if not is_final or self.final_set is FinalSet.TIEBREAK:
            return SetRule(6, 7)
        if self.final_set is FinalSet.ADVANTAGE:
            return SetRule(None, 7)
        if self.final_set is FinalSet.TIEBREAK_12:
            return SetRule(12, 7)
        return SetRule(6, 10)


@dataclass(frozen=True)
class ScoreState:
    sets: tuple[int, int] = (0, 0)
    games: tuple[int, int] = (0, 0)
    points: tuple[int, int] = (0, 0)
    server: int = 0
    tiebreak: bool = False
    # Player who served the first point of the current tiebreak (only meaningful when
    # tiebreak is True). Needed because the receiver of that first point serves first
    # in the next set.
    tiebreak_first_server: int = 0
    winner: int | None = None

    @property
    def is_over(self) -> bool:
        return self.winner is not None


def _other(player: int) -> int:
    return 1 - player


def _normalise(pts: tuple[int, int], target: int) -> tuple[int, int]:
    """Map scores like (5, 5) in a game or (8, 7) in a tiebreak to the canonical form.

    Once both players have target - 1 points, only the difference matters.
    """
    a, b = pts
    floor = target - 1
    if a >= floor and b >= floor:
        diff = a - b
        return (floor + max(diff, 0), floor + max(-diff, 0))
    return pts


def _with(t: tuple[int, int], idx: int, value: int) -> tuple[int, int]:
    return (value, t[1]) if idx == 0 else (t[0], value)


def _start_set(state: ScoreState, fmt: MatchFormat, next_server: int) -> ScoreState:
    return replace(
        state, games=(0, 0), points=(0, 0), server=next_server, tiebreak=False
    )


def _game_won(state: ScoreState, fmt: MatchFormat, w: int, last_server: int) -> ScoreState:
    """Award a (non-tiebreak) game to ``w`` and check for the end of the set."""
    games = _with(state.games, w, state.games[w] + 1)
    rule = fmt.set_rule(*state.sets)
    gw, gl = games[w], games[_other(w)]
    next_server = _other(last_server)

    if gw >= 6 and gw - gl >= 2:
        return _set_won(replace(state, games=games), fmt, w, next_server)

    state = replace(state, games=games, points=(0, 0), server=next_server)
    if rule.tiebreak_at is not None and games[0] == games[1] == rule.tiebreak_at:
        state = replace(state, tiebreak=True, tiebreak_first_server=next_server)
    return state


def _set_won(state: ScoreState, fmt: MatchFormat, w: int, next_server: int) -> ScoreState:
    sets = _with(state.sets, w, state.sets[w] + 1)
    state = replace(state, sets=sets)
    if sets[w] == fmt.sets_to_win:
        return replace(state, points=(0, 0), tiebreak=False, winner=w)
    return _start_set(state, fmt, next_server)


def advance(state: ScoreState, point_winner: int, fmt: MatchFormat) -> ScoreState:
    """Return the score after ``point_winner`` (0 or 1) wins the next point."""
    if state.is_over:
        raise ValueError("match is already over")
    if point_winner not in (0, 1):
        raise ValueError(f"point_winner must be 0 or 1, got {point_winner}")

    w = point_winner
    raw = _with(state.points, w, state.points[w] + 1)

    if state.tiebreak:
        target = fmt.set_rule(*state.sets).tiebreak_points
        if raw[w] >= target and raw[w] - raw[_other(w)] >= 2:
            games = _with(state.games, w, state.games[w] + 1)
            # The player who received first in the tiebreak serves first next set.
            nxt = _other(state.tiebreak_first_server)
            return _set_won(replace(state, games=games), fmt, w, nxt)
        # Serve changes after the 1st point and then every two points.
        played = raw[0] + raw[1]
        server = _other(state.server) if played % 2 == 1 else state.server
        return replace(state, points=_normalise(raw, target), server=server)

    if raw[w] >= 4 and raw[w] - raw[_other(w)] >= 2:
        return _game_won(state, fmt, w, last_server=state.server)
    return replace(state, points=_normalise(raw, 4))


def replay(
    winners: list[int] | tuple[int, ...], fmt: MatchFormat, first_server: int = 0
) -> list[ScoreState]:
    """Replay a sequence of point winners; returns the state *before* each point
    followed by the final state (so ``len(result) == len(winners) + 1``)."""
    state = ScoreState(server=first_server)
    states = [state]
    for w in winners:
        state = advance(state, w, fmt)
        states.append(state)
    return states


POINT_NAMES = ("0", "15", "30", "40", "AD")


def describe(state: ScoreState) -> str:
    """Human-readable score, e.g. ``'6-4 3-2 30-40 (srv B)'``."""
    if state.tiebreak:
        pts = f"TB {state.points[0]}-{state.points[1]}"
    else:
        a, b = state.points
        pts = f"{POINT_NAMES[a]}-{POINT_NAMES[b]}"
    srv = "AB"[state.server]
    sets = f"{state.sets[0]}-{state.sets[1]}"
    games = f"{state.games[0]}-{state.games[1]}"
    return f"sets {sets} games {games} {pts} (srv {srv})"
