"""Parse Betfair historical BASIC stream files for tennis MATCH_ODDS markets.

A BASIC file holds one market. Each line is a JSON ``mcm`` (market change) message:

    {"op": "mcm", "pt": <epoch ms>, "mc": [{"id": "1.234", "marketDefinition": {...},
                                             "rc": [{"id": <selection>, "ltp": 1.85}]}]}

``marketDefinition`` is sent whenever the market's metadata changes (opens, goes
in-play, suspends, settles). ``rc`` (runner changes) carries the last traded price.
BASIC data is sampled at one-minute intervals and has no volume or order book.

The output is long format, one row per (message time, runner), with prices
forward-filled so every row shows the latest known price at that moment:

    market_id, event_id, event_name, market_time, timestamp_utc, selection_id,
    runner_name, last_traded_price, in_play, market_status, runner_status

``runner_status`` is the *final* status (WINNER / LOSER / REMOVED) from the last
market definition. It is settlement information, so it must never be used as a
feature. :mod:`courtedge.market.backtest` only reads it when settling bets.

This module only reads files on disk. Nothing here talks to the Betfair API.
"""

from __future__ import annotations

import bz2
import io
import json
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

COLUMNS = [
    "market_id", "event_id", "event_name", "market_time", "timestamp_utc",
    "selection_id", "runner_name", "last_traded_price", "in_play", "market_status",
    "runner_status",
]  # fmt: skip

TENNIS_EVENT_TYPE = "2"


def _iter_lines(raw: bytes) -> Iterator[dict]:
    data = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
    for line in io.BytesIO(data):
        line = line.strip()
        if line:
            yield json.loads(line)


def parse_stream(
    messages: Iterable[dict], market_type: str = "MATCH_ODDS"
) -> pd.DataFrame:
    """Turn the ``mcm`` messages of one market into the long table described above."""
    definition: dict = {}
    market_id: str | None = None
    ltp: dict[int, float] = {}
    snapshots: list[tuple] = []  # (pt, in_play, status, {sel: ltp})

    for msg in messages:
        if msg.get("op") != "mcm":
            continue
        pt = msg["pt"]
        for mc in msg.get("mc", []):
            market_id = mc["id"]
            if "marketDefinition" in mc:
                definition = mc["marketDefinition"]
            for rc in mc.get("rc", []):
                if "ltp" in rc:
                    ltp[int(rc["id"])] = float(rc["ltp"])
            snapshots.append(
                (pt, bool(definition.get("inPlay", False)), definition.get("status"), dict(ltp))
            )

    if not definition or market_id is None:
        return pd.DataFrame(columns=COLUMNS)
    if definition.get("marketType") != market_type:
        return pd.DataFrame(columns=COLUMNS)
    if definition.get("eventTypeId", TENNIS_EVENT_TYPE) != TENNIS_EVENT_TYPE:
        return pd.DataFrame(columns=COLUMNS)

    runners = {int(r["id"]): r for r in definition.get("runners", [])}
    rows = []
    for pt, in_play, status, prices in snapshots:
        for sel, r in runners.items():
            rows.append(
                (
                    market_id, definition.get("eventId"), definition.get("eventName"),
                    definition.get("marketTime"), pt, sel, r.get("name"),
                    prices.get(sel), in_play, status, r.get("status"),
                )
            )  # fmt: skip

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
    df["market_time"] = pd.to_datetime(df["market_time"], utc=True)
    df["selection_id"] = df["selection_id"].astype("int64")
    df["last_traded_price"] = df["last_traded_price"].astype("float64")
    # Several messages can share a timestamp; keep the last state at each time.
    df = df.drop_duplicates(["timestamp_utc", "selection_id"], keep="last")
    return df.reset_index(drop=True)


def parse_file(path: str | Path, market_type: str = "MATCH_ODDS") -> pd.DataFrame:
    """Parse one BASIC file (``.bz2`` or plain newline-delimited JSON)."""
    return parse_stream(_iter_lines(Path(path).read_bytes()), market_type)


def _iter_market_blobs(root: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (name, bytes) for every market file under ``root``.

    Handles a directory tree of ``.bz2`` files (the layout after extracting a download)
    and the ``.tar`` archives that Betfair's historic data site serves.
    """
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for p in paths:
        if p.suffix == ".tar":
            with tarfile.open(p) as tar:
                for member in tar:
                    if member.isfile() and member.name.endswith(".bz2"):
                        f = tar.extractfile(member)
                        if f is not None:
                            yield member.name, f.read()
        elif p.is_file() and p.suffix == ".bz2":
            yield str(p), p.read_bytes()


def parse_path(root: str | Path, market_type: str = "MATCH_ODDS") -> pd.DataFrame:
    """Parse every market under a file, directory or ``.tar`` archive."""
    frames = [
        parse_stream(_iter_lines(blob), market_type)
        for _, blob in _iter_market_blobs(Path(root))
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (market, timestamp) with the two runners side by side.

    Runner order is the order of first appearance. Use the catalogue to map
    runners to player A and player B.
    """
    out = []
    for market_id, g in df.groupby("market_id", sort=False):
        sels = list(dict.fromkeys(g["selection_id"]))
        if len(sels) != 2:
            continue  # walkovers / malformed markets
        piv = g.pivot(index="timestamp_utc", columns="selection_id", values="last_traded_price")
        meta = g.drop_duplicates("timestamp_utc").set_index("timestamp_utc")
        wide = pd.DataFrame(
            {
                "market_id": market_id,
                "ltp_1": piv[sels[0]],
                "ltp_2": piv[sels[1]],
                "in_play": meta["in_play"],
                "market_status": meta["market_status"],
            }
        )
        wide["selection_1"], wide["selection_2"] = sels
        out.append(wide.reset_index())
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)
