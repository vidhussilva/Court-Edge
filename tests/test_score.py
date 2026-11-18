import pytest

from courtedge.markov.score import FinalSet, MatchFormat, ScoreState, advance, describe, replay

BO3 = MatchFormat(3, FinalSet.TIEBREAK)


def play(state, winners, fmt=BO3):
    for w in winners:
        state = advance(state, w, fmt)
    return state


def test_love_game_switches_server():
    s = play(ScoreState(server=0), [0, 0, 0, 0])
    assert s.games == (1, 0) and s.points == (0, 0) and s.server == 1


def test_deuce_is_normalised():
    s = play(ScoreState(), [0, 1] * 5)  # 5-5 in raw points
    assert s.points == (3, 3)
    assert play(s, [0]).points == (4, 3)
    assert play(s, [0, 1]).points == (3, 3)
    assert play(s, [1, 1]).games == (0, 1)


def test_tiebreak_serve_rotation_and_next_set_server():
    s = ScoreState(games=(6, 5), server=1)
    s = play(s, [1, 1, 1, 1])  # B holds -> 6-6, A serves first in the tiebreak
    assert s.tiebreak and s.server == 0 and s.tiebreak_first_server == 0
    servers = []
    for w in [0, 1, 0, 1, 0, 1]:
        servers.append(s.server)
        s = advance(s, w, BO3)
    assert servers == [0, 1, 1, 0, 0, 1]  # A, then B B, then A A, ...
    s = play(s, [0, 0, 0, 0, 0])  # from 3-3 A wins 7-3
    assert s.sets == (1, 0) and s.games == (0, 0)
    assert s.server == 1  # tiebreak receiver serves first in the next set


def test_long_tiebreak_needs_two_clear():
    s = ScoreState(games=(6, 6), tiebreak=True, server=0)
    s = play(s, [0, 1] * 8)  # 8-8
    assert s.tiebreak and s.points == (6, 6)
    s = play(s, [0])
    assert s.tiebreak
    s = play(s, [0])
    assert s.sets == (1, 0)


def test_set_ends_7_5_without_tiebreak():
    s = ScoreState(games=(6, 5), server=0)
    s = play(s, [0] * 4)
    assert s.sets == (1, 0)


@pytest.mark.parametrize(
    "rule, tb_at, tb_pts",
    [
        (FinalSet.TIEBREAK, 6, 7),
        (FinalSet.SUPER_TIEBREAK, 6, 10),
        (FinalSet.TIEBREAK_12, 12, 7),
        (FinalSet.ADVANTAGE, None, 7),
    ],
)
def test_final_set_rules(rule, tb_at, tb_pts):
    fmt = MatchFormat(5, rule)
    assert fmt.set_rule(2, 2).tiebreak_at == tb_at
    assert fmt.set_rule(2, 2).tiebreak_points == tb_pts
    assert fmt.set_rule(2, 1).tiebreak_at == 6  # non-deciding sets are always standard


def test_advantage_final_set_runs_long():
    fmt = MatchFormat(3, FinalSet.ADVANTAGE)
    s = ScoreState(sets=(1, 1), games=(6, 6), server=0)
    s = play(s, [0] * 4 + [1] * 4, fmt)  # both hold: 7-7
    assert not s.tiebreak and s.games == (7, 7)


def test_super_tiebreak_to_ten():
    fmt = MatchFormat(5, FinalSet.SUPER_TIEBREAK)
    s = ScoreState(sets=(2, 2), games=(6, 6), tiebreak=True, server=0)
    s = play(s, [0] * 7, fmt)
    assert not s.is_over
    s = play(s, [0] * 3, fmt)
    assert s.winner == 0 and s.sets == (3, 2)


def test_match_over_raises():
    s = ScoreState(sets=(1, 0), games=(5, 0), server=0)
    s = play(s, [0] * 4)
    assert s.winner == 0
    with pytest.raises(ValueError):
        advance(s, 0, BO3)


def test_replay_lengths_and_describe():
    states = replay([0] * 48, BO3)
    assert len(states) == 49 and states[-1].winner == 0
    assert "sets 0-0" in describe(states[0])
