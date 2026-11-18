"""Link each point-by-point match to its Betfair MATCH_ODDS market.

The two sources share no IDs, so matching uses:

* names: Betfair runner names ("Carlos Alcaraz") against Sackmann player names.
  Surnames are compared after stripping accents and punctuation, with a fuzzy
  ratio to absorb transliteration differences (Medvedev / Medvedev, Wawrinka /
  Wawrinka S). Both players must match, in either order.
* dates: ``tennis_pointbypoint`` rows carry a date (allow ±2 days for timezones and
  rain delays). ``tennis_slam_pointbypoint`` rows only carry year and slam, so the
  market date must fall in that slam's months.

``match_confidence`` = min(surname similarity A, surname similarity B) × date factor.
Assignment is greedy and one-to-one, best confidence first. Pairs below
``min_confidence`` are reported as unmatched rather than forced.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

SLAM_MONTHS = {
    "ausopen": {1, 2},
    "frenchopen": {5, 6},
    "wimbledon": {6, 7},
    "usopen": {8, 9},
}

CATALOGUE_COLUMNS = [
    "match_id", "source", "tournament", "date", "player_a", "player_b", "best_of",
    "final_set", "market_id", "event_name", "selection_a", "selection_b", "market_time",
    "inplay_start_utc", "market_end_utc", "match_confidence", "video_path", "video_offset_s",
]  # fmt: skip


def _ascii(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def surname(name: str) -> str:
    """Last token of a name; handles 'Del Potro', 'de Minaur', 'Auger-Aliassime'."""
    tokens = _ascii(name).split()
    if not tokens:
        return ""
    if len(tokens) >= 3 and tokens[-2] in {"del", "de", "van", "von", "da", "di", "der"}:
        return " ".join(tokens[-2:])
    return tokens[-1]


def name_similarity(a: str, b: str) -> float:
    sa, sb = surname(a), surname(b)
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0
    return SequenceMatcher(None, sa, sb).ratio()


def _date_factor(pbp_row: pd.Series, market_time: pd.Timestamp) -> float:
    if pd.notna(pbp_row.get("date")):
        days = abs((market_time.tz_localize(None).normalize() - pd.Timestamp(pbp_row["date"])).days)
        return {0: 1.0, 1: 0.98, 2: 0.95}.get(days, 0.0)
    months = SLAM_MONTHS.get(str(pbp_row.get("tournament", "")).lower())
    if months is None or pbp_row.get("year") != market_time.year:
        return 0.0
    return 1.0 if market_time.month in months else 0.0


def market_summary(betfair_long: pd.DataFrame) -> pd.DataFrame:
    """One row per market: runners, scheduled time, in-play start, and end time."""
    rows = []
    for market_id, g in betfair_long.groupby("market_id", sort=False):
        runners = g.drop_duplicates("selection_id")[["selection_id", "runner_name"]]
        if len(runners) != 2:
            continue
        ip = g.loc[g["in_play"], "timestamp_utc"]
        inplay_start = ip.min() if len(ip) else pd.NaT
        after = g[(g["timestamp_utc"] >= inplay_start) & g["market_status"].isin(
            ["SUSPENDED", "CLOSED"])] if len(ip) else g.iloc[0:0]
        end = after["timestamp_utc"].min() if len(after) else g["timestamp_utc"].max()
        rows.append(
            dict(
                market_id=market_id, event_name=g["event_name"].iloc[0],
                market_time=g["market_time"].iloc[0],
                sel_1=int(runners.iloc[0, 0]), name_1=runners.iloc[0, 1],
                sel_2=int(runners.iloc[1, 0]), name_2=runners.iloc[1, 1],
                inplay_start_utc=inplay_start, market_end_utc=end,
            )
        )  # fmt: skip
    return pd.DataFrame(rows)


def build_catalogue(
    matches: pd.DataFrame, markets: pd.DataFrame, min_confidence: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (catalogue, unmatched_matches)."""
    candidates = []
    for mi, m in matches.iterrows():
        for ki, k in markets.iterrows():
            df = _date_factor(m, k["market_time"])
            if df == 0.0:
                continue
            straight = min(name_similarity(m.player_a, k.name_1),
                           name_similarity(m.player_b, k.name_2))  # fmt: skip
            swapped = min(name_similarity(m.player_a, k.name_2),
                          name_similarity(m.player_b, k.name_1))  # fmt: skip
            sim, flip = (straight, False) if straight >= swapped else (swapped, True)
            conf = sim * df
            if conf >= min_confidence:
                candidates.append((conf, mi, ki, flip))

    used_m, used_k, rows = set(), set(), []
    for conf, mi, ki, flip in sorted(candidates, key=lambda c: -c[0]):
        if mi in used_m or ki in used_k:
            continue
        used_m.add(mi)
        used_k.add(ki)
        m, k = matches.loc[mi], markets.loc[ki]
        sel_a, sel_b = (k.sel_2, k.sel_1) if flip else (k.sel_1, k.sel_2)
        rows.append(
            dict(
                match_id=m.match_id, source=m.source, tournament=m.tournament,
                date=k.market_time.date(), player_a=m.player_a, player_b=m.player_b,
                best_of=m.best_of, final_set=m.final_set, market_id=k.market_id,
                event_name=k.event_name, selection_a=sel_a, selection_b=sel_b,
                market_time=k.market_time, inplay_start_utc=k.inplay_start_utc,
                market_end_utc=k.market_end_utc, match_confidence=round(conf, 3),
                video_path="", video_offset_s=pd.NA,
            )
        )  # fmt: skip
    catalogue = pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)
    unmatched = matches.loc[[i for i in matches.index if i not in used_m]]
    return catalogue, unmatched


def load_catalogue(path) -> pd.DataFrame:
    """Read catalogue.csv with the right dtypes (market ids look like floats: '1.216')."""
    return pd.read_csv(
        path,
        dtype={"market_id": str, "match_id": str, "video_path": str},
        parse_dates=["market_time", "inplay_start_utc", "market_end_utc"],
        keep_default_na=True,
    ).fillna({"video_path": ""})
