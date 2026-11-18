from pathlib import Path

import pandas as pd
import pytest

from courtedge.data import betfair, pbp
from courtedge.data.catalogue import (
    build_catalogue,
    load_catalogue,
    market_summary,
    name_similarity,
    surname,
)

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Juan Martin Del Potro", "del potro"),
        ("Alex de Minaur", "de minaur"),
        ("Félix Auger-Aliassime", "aliassime"),
        ("Novak Djokovic", "djokovic"),
    ],
)
def test_surname(name, expected):
    assert surname(name) == expected


def test_name_similarity_accents_and_spelling():
    assert name_similarity("Stan Wawrinka", "Stanislas Wawrinka") == 1.0
    assert name_similarity("Daniil Medvedev", "D Medvedev") == 1.0
    assert name_similarity("Novak Djokovic", "Andy Murray") < 0.5


def _sources():
    _, matches = pbp.load_slam(FIX / "slam" / "2023-wimbledon-points.csv")
    markets = market_summary(betfair.parse_path(FIX / "betfair"))
    return matches, markets


def test_links_fixture_match_and_orientation():
    matches, markets = _sources()
    cat, unmatched = build_catalogue(matches, markets)
    assert len(cat) == 1 and len(unmatched) == 1
    row = cat.iloc[0]
    assert row.match_id == "2023-wimbledon-1101"
    assert row.selection_a == 2249229  # Alcaraz is player 1 in the fixture
    assert row.match_confidence == 1.0
    assert row.inplay_start_utc < row.market_end_utc


def test_swapped_runner_order_is_detected():
    matches, markets = _sources()
    m = markets.copy()
    m[["sel_1", "name_1", "sel_2", "name_2"]] = m[["sel_2", "name_2", "sel_1", "name_1"]].to_numpy()
    cat, _ = build_catalogue(matches, m)
    assert cat.iloc[0].selection_a == 2249229


def test_wrong_month_is_not_linked():
    matches, markets = _sources()
    m = markets.copy()
    m["market_time"] = m["market_time"] + pd.Timedelta(days=90)
    cat, _ = build_catalogue(matches, m)
    assert cat.empty


def test_catalogue_csv_roundtrip(tmp_path):
    matches, markets = _sources()
    cat, _ = build_catalogue(matches, markets)
    cat.to_csv(tmp_path / "c.csv", index=False)
    back = load_catalogue(tmp_path / "c.csv")
    assert back.market_id.iloc[0] == "1.216000001"
    assert back.inplay_start_utc.dt.tz is not None
