from pathlib import Path

import pandas as pd
import pytest

from courtedge.data import pbp
from courtedge.markov.score import FinalSet, MatchFormat, advance

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def slam():
    return pbp.load_slam(FIX / "slam" / "2023-wimbledon-points.csv")


def test_slam_parses_cleanly(slam):
    points, matches = slam
    assert len(matches) == 2
    assert (matches["parse_issues"] == "").all(), matches["parse_issues"].tolist()
    men = matches.set_index("match_id").loc["2023-wimbledon-1101"]
    assert men.best_of == 5 and men.final_set == "super_tiebreak"
    women = matches.set_index("match_id").loc["2023-wimbledon-2101"]
    assert women.best_of == 3


def test_slam_points_are_tidy(slam):
    points, matches = slam
    assert list(points.columns) == pbp.POINT_COLUMNS
    for _, g in points.groupby("match_id"):
        assert g["point_index"].tolist() == list(range(len(g)))
        assert g["elapsed_s"].is_monotonic_increasing
    first = points.groupby("match_id").head(1)
    score_cols = ["sets_a", "sets_b", "games_a", "games_b", "points_a", "points_b"]
    assert (first[score_cols] == 0).all().all()
    assert points["tiebreak"].any()


def test_state_at_roundtrip(slam):
    points, matches = slam
    g = points[points.match_id == "2023-wimbledon-1101"].reset_index(drop=True)
    fmt = MatchFormat(5, FinalSet.SUPER_TIEBREAK)
    for i in range(len(g) - 1):
        nxt = advance(pbp.state_at(g.iloc[i]), int(g.loc[i, "winner"]), fmt)
        assert nxt == pbp.state_at(g.iloc[i + 1])


def test_slam_detects_server_mismatch(tmp_path):
    src = FIX / "slam"
    df = pd.read_csv(src / "2023-wimbledon-points.csv", dtype=str)
    row = df.index[(df.match_id == "2023-wimbledon-2101") & (df.PointNumber == "5")][0]
    df.loc[row, "PointServer"] = "2" if df.loc[row, "PointServer"] == "1" else "1"
    df.to_csv(tmp_path / "2023-wimbledon-points.csv", index=False)
    matches_csv = (src / "2023-wimbledon-matches.csv").read_text()
    (tmp_path / "2023-wimbledon-matches.csv").write_text(matches_csv)
    _, matches = pbp.load_slam(tmp_path / "2023-wimbledon-points.csv")
    issues = matches.set_index("match_id").loc["2023-wimbledon-2101", "parse_issues"]
    assert "server mismatch on 1 points" in issues


@pytest.mark.parametrize(
    "slam, year, rule",
    [
        ("wimbledon", 2018, FinalSet.ADVANTAGE),
        ("wimbledon", 2019, FinalSet.TIEBREAK_12),
        ("ausopen", 2019, FinalSet.SUPER_TIEBREAK),
        ("frenchopen", 2021, FinalSet.ADVANTAGE),
        ("usopen", 2015, FinalSet.TIEBREAK),
        ("usopen", 2023, FinalSet.SUPER_TIEBREAK),
    ],
)
def test_final_set_rule_table(slam, year, rule):
    assert pbp.slam_final_set_rule(slam, year) is rule


def test_pointbypoint_format():
    points, matches = pbp.load_pointbypoint(FIX / "pbp_matches_atp_main_current.csv")
    assert len(matches) == 2
    assert (matches["parse_issues"] == "").all(), matches["parse_issues"].tolist()
    assert (matches["first_server"] == 0).all()
    assert matches["date"].notna().all()
    assert points.groupby("match_id")["server"].first().eq(0).all()


def test_parse_pbp_string():
    won, ends = pbp.parse_pbp_string("SSSS;RRRR.A/D")
    assert won == [True] * 4 + [False] * 4 + [True, False]
    assert ends == [3, 7]
    with pytest.raises(ValueError):
        pbp.parse_pbp_string("SSX")
