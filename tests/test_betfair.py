from pathlib import Path

from courtedge.data import betfair

FIX = Path(__file__).parent / "fixtures" / "betfair"


def test_parse_file_columns_and_types():
    df = betfair.parse_file(FIX / "1.216000001.bz2")
    assert list(df.columns) == betfair.COLUMNS
    assert df["market_id"].unique().tolist() == ["1.216000001"]
    assert df["selection_id"].nunique() == 2
    assert df["timestamp_utc"].dt.tz is not None
    assert df["timestamp_utc"].is_monotonic_increasing


def test_inplay_flag_and_settlement():
    df = betfair.parse_file(FIX / "1.216000001.bz2")
    assert not df["in_play"].iloc[0] and df["in_play"].iloc[-1]
    assert df["market_status"].iloc[-1] == "CLOSED"
    final = df.drop_duplicates("selection_id")["runner_status"]
    assert sorted(final) == ["LOSER", "WINNER"]


def test_prices_forward_filled():
    df = betfair.parse_file(FIX / "1.216000001.bz2")
    # The in-play and settlement messages carry no prices; rows still show the last one.
    assert df["last_traded_price"].notna().all()


def test_parse_path_handles_tar(tmp_path):
    import tarfile

    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(FIX / "1.216000001.bz2", arcname="BASIC/2023/Jul/16/32500001/1.216000001.bz2")
    df = betfair.parse_path(tar_path)
    assert df["market_id"].nunique() == 1


def _msg(pt, market_type="MATCH_ODDS", event_type="2"):
    return {
        "op": "mcm",
        "pt": pt,
        "mc": [
            {
                "id": "1.1",
                "marketDefinition": {
                    "marketType": market_type,
                    "eventTypeId": event_type,
                    "status": "OPEN",
                    "inPlay": False,
                    "runners": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                },
                "rc": [{"id": 1, "ltp": 1.5}, {"id": 2, "ltp": 2.9}],
            }
        ],
    }


def test_filters_other_markets_and_sports():
    assert betfair.parse_stream([_msg(0, market_type="SET_BETTING")]).empty
    assert betfair.parse_stream([_msg(0, event_type="1")]).empty
    assert len(betfair.parse_stream([_msg(0)])) == 2


def test_to_wide():
    wide = betfair.to_wide(betfair.parse_file(FIX / "1.216000001.bz2"))
    assert {"ltp_1", "ltp_2", "selection_1", "selection_2", "in_play"} <= set(wide.columns)
    assert wide["timestamp_utc"].is_unique
