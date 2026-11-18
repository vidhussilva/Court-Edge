"""Load Jeff Sackmann's point-by-point data into one tidy table: one row per point.

Two upstream formats are supported:

* ``tennis_slam_pointbypoint`` -- ``<year>-<slam>-points.csv`` with one row per point
  (PointServer, PointWinner, ElapsedTime, ...) plus ``<year>-<slam>-matches.csv``
  with player names.
* ``tennis_pointbypoint`` -- one row per match with the whole match in a ``pbp``
  string: ``S``/``A`` server wins the point, ``R``/``D`` returner wins, ``;`` ends a
  game, ``.`` ends a set, ``/`` marks a change of server inside a tiebreak.

In both cases the score is rebuilt with :mod:`courtedge.markov.score` rather than
trusting the upstream score columns. The upstream columns are only used to check
the result. Any disagreement is recorded in ``matches.parse_issues``, not hidden.

Output schema
-------------
points:  match_id, point_index, server, winner, sets_a, sets_b, games_a, games_b,
         points_a, points_b, tiebreak, tb_first_server, elapsed_s
         (the score columns describe the state *before* the point)
matches: match_id, source, date, year, tournament, player_a, player_b, best_of,
         final_set, first_server, winner, n_points, parse_issues
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from courtedge.markov.score import FinalSet, MatchFormat, ScoreState, advance

POINT_COLUMNS = [
    "match_id", "point_index", "server", "winner",
    "sets_a", "sets_b", "games_a", "games_b", "points_a", "points_b",
    "tiebreak", "tb_first_server", "elapsed_s",
]  # fmt: skip


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def slam_final_set_rule(slam: str, year: int) -> FinalSet:
    """Deciding-set rule used at each Grand Slam in a given year."""
    slam = slam.lower()
    if year >= 2022:
        return FinalSet.SUPER_TIEBREAK  # 10-point tiebreak at 6-6 at all four slams
    if slam == "usopen":
        return FinalSet.TIEBREAK
    if slam == "wimbledon" and year >= 2019:
        return FinalSet.TIEBREAK_12
    if slam == "ausopen" and year >= 2019:
        return FinalSet.SUPER_TIEBREAK
    return FinalSet.ADVANTAGE


def _state_row(match_id: str, i: int, s: ScoreState, winner: int, elapsed: float) -> tuple:
    return (
        match_id, i, s.server, winner,
        s.sets[0], s.sets[1], s.games[0], s.games[1], s.points[0], s.points[1],
        s.tiebreak, s.tiebreak_first_server, elapsed,
    )  # fmt: skip


def _replay_rows(
    match_id: str,
    winners: Iterable[int],
    fmt: MatchFormat,
    first_server: int,
    elapsed: Iterable[float] | None = None,
    servers: Iterable[int] | None = None,
) -> tuple[list[tuple], ScoreState, list[str]]:
    """Replay a match and collect one row per point plus any consistency issues."""
    winners = list(winners)
    elapsed = list(elapsed) if elapsed is not None else [np.nan] * len(winners)
    servers = list(servers) if servers is not None else None
    issues: list[str] = []
    rows: list[tuple] = []
    state = ScoreState(server=first_server)
    server_mismatches = 0
    for i, w in enumerate(winners):
        if state.is_over:
            issues.append(f"{len(winners) - i} points after match end")
            break
        if servers is not None and servers[i] != state.server:
            server_mismatches += 1
        rows.append(_state_row(match_id, i, state, w, elapsed[i]))
        state = advance(state, w, fmt)
    if server_mismatches:
        issues.append(f"server mismatch on {server_mismatches} points")
    if not state.is_over:
        issues.append("match incomplete (retirement or missing points)")
    return rows, state, issues


def _to_frames(rows: list[tuple], matches: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.DataFrame(rows, columns=POINT_COLUMNS)
    int_cols = [c for c in POINT_COLUMNS if c not in ("match_id", "tiebreak", "elapsed_s")]
    points[int_cols] = points[int_cols].astype("int16")
    points["point_index"] = points["point_index"].astype("int32")
    points["elapsed_s"] = points["elapsed_s"].astype("float64")
    return points, pd.DataFrame(matches)


# ---------------------------------------------------------------------------
# tennis_slam_pointbypoint
# ---------------------------------------------------------------------------

_SLAM_FILE = re.compile(r"(?P<year>\d{4})-(?P<slam>[a-z]+)-points\.csv$")


def _parse_elapsed(value: object) -> float:
    """'1:02:33' -> 3753.0. Missing or malformed values become NaN."""
    if not isinstance(value, str) or ":" not in value:
        return np.nan
    try:
        h, m, s = (int(x) for x in value.split(":"))
    except ValueError:
        return np.nan
    return float(h * 3600 + m * 60 + s)


def load_slam(points_csv: str | Path, matches_csv: str | Path | None = None):
    """Parse one ``<year>-<slam>-points.csv`` (and its matches file) into tidy frames."""
    points_csv = Path(points_csv)
    m = _SLAM_FILE.search(points_csv.name)
    if m is None:
        raise ValueError(f"unexpected slam file name: {points_csv.name}")
    year, slam = int(m["year"]), m["slam"]
    if matches_csv is None:
        matches_csv = points_csv.with_name(f"{year}-{slam}-matches.csv")

    raw = pd.read_csv(points_csv, dtype={"PointNumber": str}, low_memory=False)
    meta = pd.read_csv(matches_csv).set_index("match_id")

    # Some files start each match with placeholder rows ("0X", "0Y") and rows where
    # nobody won the point. Keep real points only.
    raw = raw[pd.to_numeric(raw["PointNumber"], errors="coerce").fillna(0) > 0]
    raw = raw[raw["PointWinner"].isin([1, 2])].copy()
    raw["PointNumber"] = raw["PointNumber"].astype(int)
    final_rule = slam_final_set_rule(slam, year)

    all_rows: list[tuple] = []
    matches: list[dict] = []
    for match_id, g in raw.groupby("match_id", sort=False):
        g = g.sort_values("PointNumber")
        if match_id not in meta.index:
            continue
        info = meta.loc[match_id]
        match_num = int(str(match_id).rsplit("-", 1)[-1])
        if match_num // 1000 not in (1, 2):  # 1xxx men's singles, 2xxx women's singles
            continue
        best_of = 5 if match_num // 1000 == 1 else 3
        fmt = MatchFormat(best_of, final_rule)

        winners = (g["PointWinner"].to_numpy() - 1).tolist()
        servers = (g["PointServer"].to_numpy() - 1).tolist()
        elapsed = [_parse_elapsed(v) for v in g.get("ElapsedTime", [None] * len(g))]
        rows, final, issues = _replay_rows(
            str(match_id), winners, fmt, servers[0], elapsed, servers
        )
        upstream_winner = info.get("winner")
        if final.is_over and pd.notna(upstream_winner) and int(upstream_winner) - 1 != final.winner:
            issues.append("winner disagrees with matches file")

        all_rows.extend(rows)
        matches.append(
            dict(
                match_id=str(match_id), source="slam", date=pd.NaT, year=year,
                tournament=slam, player_a=info["player1"], player_b=info["player2"],
                best_of=best_of, final_set=final_rule.value, first_server=servers[0],
                winner=final.winner, n_points=len(rows), parse_issues="; ".join(issues),
            )
        )  # fmt: skip
    return _to_frames(all_rows, matches)


# ---------------------------------------------------------------------------
# tennis_pointbypoint
# ---------------------------------------------------------------------------


def parse_pbp_string(pbp: str) -> tuple[list[bool], list[int]]:
    """Split a pbp string into per-point "server won?" flags and the index of the
    last point of every game (taken from the ``;`` and ``.`` separators)."""
    server_won: list[bool] = []
    game_ends: list[int] = []
    for ch in pbp:
        if ch in "SA":
            server_won.append(True)
        elif ch in "RD":
            server_won.append(False)
        elif ch in ";.":
            if server_won and (not game_ends or game_ends[-1] != len(server_won) - 1):
                game_ends.append(len(server_won) - 1)
        elif ch in "/ ":
            continue
        else:
            raise ValueError(f"unexpected character {ch!r} in pbp string")
    return server_won, game_ends


def _infer_best_of(score: object) -> int:
    """'6-4 3-6 6-7(5) 6-3 6-2' -> 5. Falls back to 3 when the score is unusable."""
    if not isinstance(score, str):
        return 3
    sets = re.findall(r"(\d+)-(\d+)", score)
    won = [0, 0]
    for a, b in sets:
        won[int(a) < int(b)] += 1
    return 5 if max(won) >= 3 else 3


def load_pointbypoint(csv_path: str | Path, tour: str | None = None):
    """Parse a ``tennis_pointbypoint`` CSV (e.g. ``pbp_matches_atp_main_current.csv``).

    Player A is ``server1`` (the player who served first), so ``first_server`` is 0.
    """
    raw = pd.read_csv(csv_path)
    all_rows: list[tuple] = []
    matches: list[dict] = []
    for rec in raw.itertuples(index=False):
        match_id = f"pbp-{rec.pbp_id}"
        best_of = _infer_best_of(getattr(rec, "score", None))
        fmt = MatchFormat(best_of, FinalSet.TIEBREAK)
        try:
            server_won, game_ends = parse_pbp_string(str(rec.pbp))
        except ValueError as exc:
            matches.append(dict(match_id=match_id, source="pbp", parse_issues=str(exc)))
            continue

        # The point winner depends on who is serving, and that comes from replaying
        # the score. So we replay here instead of calling _replay_rows.
        rows: list[tuple] = []
        issues: list[str] = []
        state = ScoreState(server=0)
        replay_game_ends: list[int] = []
        for i, sw in enumerate(server_won):
            if state.is_over:
                issues.append(f"{len(server_won) - i} points after match end")
                break
            w = state.server if sw else 1 - state.server
            rows.append(_state_row(match_id, i, state, w, np.nan))
            before = (state.sets, state.games)
            state = advance(state, w, fmt)
            if (state.sets, state.games) != before:
                replay_game_ends.append(i)
        # The last game usually has no trailing separator, so allow one unmarked game.
        n = min(len(replay_game_ends), len(game_ends))
        if replay_game_ends[:n] != game_ends[:n] or len(replay_game_ends) - len(game_ends) > 1:
            issues.append("game boundaries disagree with separators")
        if not state.is_over:
            issues.append("match incomplete (retirement or missing points)")
        upstream_winner = getattr(rec, "winner", None)
        if state.is_over and upstream_winner in (1, 2) and upstream_winner - 1 != state.winner:
            issues.append("winner disagrees with upstream")

        all_rows.extend(rows)
        date = pd.to_datetime(rec.date, format="%d %b %y", errors="coerce")
        matches.append(
            dict(
                match_id=match_id, source="pbp", date=date,
                year=date.year if pd.notna(date) else None, tournament=rec.tny_name,
                player_a=rec.server1, player_b=rec.server2, best_of=best_of,
                final_set=FinalSet.TIEBREAK.value, first_server=0, winner=state.winner,
                n_points=len(rows), parse_issues="; ".join(issues),
            )
        )  # fmt: skip
    points, match_df = _to_frames(all_rows, matches)
    if tour is not None:
        match_df["tour"] = tour
    return points, match_df


def state_at(row: pd.Series | dict) -> ScoreState:
    """Rebuild a :class:`ScoreState` from one row of the points table."""
    return ScoreState(
        sets=(int(row["sets_a"]), int(row["sets_b"])),
        games=(int(row["games_a"]), int(row["games_b"])),
        points=(int(row["points_a"]), int(row["points_b"])),
        server=int(row["server"]),
        tiebreak=bool(row["tiebreak"]),
        tiebreak_first_server=int(row["tb_first_server"]),
    )
