import pandas as pd
import pytest

from courtedge.data import atp


def _matches(rows):
    """rows: (date, level, surface, winner, loser, w_svpt, w_won, l_svpt, l_won)"""
    out = []
    for d, lvl, surf, w, l_, wsv, wwon, lsv, lwon in rows:
        out.append(dict(tourney_date=pd.Timestamp(d), tourney_level=lvl, surface=surf,
                        winner_name=w, loser_name=l_, w_svpt=wsv, w_1stWon=wwon, w_2ndWon=0,
                        l_svpt=lsv, l_1stWon=lwon, l_2ndWon=0))  # fmt: skip
    return pd.DataFrame(out)


BASE = [
    ("2023-01-02", "A", "Hard", "Ann Alpha", "Bob Beta", 80, 56, 80, 48),
    ("2023-02-06", "A", "Hard", "Ann Alpha", "Cid Gamma", 70, 49, 75, 45),
    ("2023-03-06", "A", "Clay", "Bob Beta", "Cid Gamma", 90, 55, 85, 50),
]


def test_no_future_matches_used():
    long = atp.to_long(_matches(BASE))
    before = atp.estimate(long, "Ann Alpha", "Bob Beta", pd.Timestamp("2023-06-01"), "Hard")
    future = BASE + [
        ("2023-06-05", "A", "Hard", "Bob Beta", "Ann Alpha", 100, 95, 100, 20),
        # Same slam as the target match: started before as_of but not finished.
        ("2023-05-29", "G", "Hard", "Bob Beta", "Ann Alpha", 100, 95, 100, 20),
    ]
    long2 = atp.to_long(_matches(future))
    after = atp.estimate(long2, "Ann Alpha", "Bob Beta", pd.Timestamp("2023-06-01"), "Hard")
    assert before == after


def test_finished_tournament_is_used():
    rows = BASE + [("2023-05-01", "A", "Hard", "Bob Beta", "Ann Alpha", 100, 95, 100, 20)]
    long = atp.to_long(_matches(rows))
    est0 = atp.estimate(atp.to_long(_matches(BASE)), "Ann Alpha", "Bob Beta",
                        pd.Timestamp("2023-06-01"), "Hard")  # fmt: skip
    est1 = atp.estimate(long, "Ann Alpha", "Bob Beta", pd.Timestamp("2023-06-01"), "Hard")
    assert est1.pb > est0.pb and est1.pa < est0.pa


def test_unknown_players_get_tour_average():
    long = atp.to_long(_matches(BASE))
    est = atp.estimate(long, "Nobody One", "Nobody Two", pd.Timestamp("2023-06-01"), "Hard")
    assert est.pa == pytest.approx(est.tour_serve_rate)
    assert est.pa == pytest.approx(est.pb)


def test_window_excludes_old_matches():
    long = atp.to_long(_matches(BASE))
    est = atp.estimate(long, "Ann Alpha", "Bob Beta", pd.Timestamp("2025-01-01"), "Hard")
    assert est.serve_points_a == 0


def test_opponent_adjustment_direction():
    # Cid is a strong returner (wins many points against serve): facing him lowers pa.
    rows = BASE + [("2023-04-03", "A", "Hard", "Cid Gamma", "Dan Delta", 60, 40, 80, 30)] * 6
    long = atp.to_long(_matches(rows))
    as_of = pd.Timestamp("2023-06-01")
    vs_bob = atp.estimate(long, "Ann Alpha", "Bob Beta", as_of, "Hard").pa
    vs_cid = atp.estimate(long, "Ann Alpha", "Cid Gamma", as_of, "Hard").pa
    assert vs_cid < vs_bob


def test_name_normalisation():
    assert atp.norm_name("Félix Auger-Aliassime") == "felix auger aliassime"
