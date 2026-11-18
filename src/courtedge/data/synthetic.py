"""Synthetic matches written in the *exact* upstream file formats.

Used for two things:

1. Test fixtures. The parsers are tested on files that look byte-for-byte like
   Sackmann's CSVs and Betfair's BASIC stream files.
2. A clearly labelled demo dataset, so the whole pipeline (catalogue, backtest,
   report) can run before the real Betfair files are downloaded.

Nothing produced here is ever reported as a real result. Reports built from it say
SYNTHETIC in their title.
"""

from __future__ import annotations

import bz2
import csv
import json
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from courtedge.market.ticks import to_tick
from courtedge.markov.score import MatchFormat, ScoreState, advance

POINT_NAMES = ("0", "15", "30", "40", "AD")


@dataclass
class SimulatedMatch:
    match_id: str
    player_a: str
    player_b: str
    fmt: MatchFormat
    first_server: int
    winners: list[int] = field(default_factory=list)
    servers: list[int] = field(default_factory=list)
    states_after: list[ScoreState] = field(default_factory=list)
    elapsed_s: list[float] = field(default_factory=list)  # time each point *ended*
    aces: list[bool] = field(default_factory=list)
    double_faults: list[bool] = field(default_factory=list)

    @property
    def winner(self) -> int:
        return self.states_after[-1].winner  # type: ignore[return-value]

    @property
    def duration_s(self) -> float:
        return self.elapsed_s[-1] if self.elapsed_s else 0.0


def simulate_match(
    match_id: str,
    player_a: str,
    player_b: str,
    pa: float,
    pb: float,
    fmt: MatchFormat,
    rng: np.random.Generator,
    first_server: int | None = None,
) -> SimulatedMatch:
    """Play a match point by point. ``pa``/``pb`` = P(point won on own serve)."""
    if first_server is None:
        first_server = int(rng.integers(2))
    m = SimulatedMatch(match_id, player_a, player_b, fmt, first_server)
    state = ScoreState(server=first_server)
    t = 0.0
    while not state.is_over:
        s = state.server
        p = pa if s == 0 else pb
        w = s if rng.random() < p else 1 - s
        # Realistic-ish timing: 25-55 s per point incl. the gap before the next serve,
        # 90 s at changeovers (after odd-numbered games) and 120 s between sets.
        t += float(rng.uniform(25, 55))
        new = advance(state, w, fmt)
        if new.sets != state.sets and not new.is_over:
            t += 120.0
        elif new.games != state.games and sum(new.games) % 2 == 1:
            t += 90.0
        m.winners.append(w)
        m.servers.append(s)
        m.states_after.append(new)
        m.elapsed_s.append(t)
        m.aces.append(w == s and rng.random() < 0.08)
        m.double_faults.append(w != s and rng.random() < 0.06)
        state = new
    return m


# ---------------------------------------------------------------------------
# Sackmann tennis_slam_pointbypoint
# ---------------------------------------------------------------------------

SLAM_POINT_HEADER = [
    "match_id", "ElapsedTime", "SetNo", "P1GamesWon", "P2GamesWon", "SetWinner",
    "GameNo", "GameWinner", "PointNumber", "PointWinner", "PointServer", "Speed_KMH",
    "Rally", "P1Score", "P2Score", "P1Ace", "P2Ace", "P1DoubleFault", "P2DoubleFault",
]  # fmt: skip
SLAM_MATCH_HEADER = [
    "match_id", "year", "slam", "match_num", "player1", "player2", "status", "winner",
    "event_name", "round", "court_name", "court_id", "player1id", "player2id",
    "nation1", "nation2",
]  # fmt: skip


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def write_slam_files(
    out_dir: Path, year: int, slam: str, matches: list[SimulatedMatch]
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pts_path = out_dir / f"{year}-{slam}-points.csv"
    m_path = out_dir / f"{year}-{slam}-matches.csv"
    with pts_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(SLAM_POINT_HEADER)
        for m in matches:
            # Upstream files often start with placeholder rows; reproduce that.
            wr.writerow([m.match_id, "0:00:00", 1, 0, 0, 0, 1, 0, "0X", 0, 0, "", "", 0, 0,
                         0, 0, 0, 0])  # fmt: skip
            set_no, game_no = 1, 1
            prev = ScoreState(server=m.first_server)
            for i, (w, s, st) in enumerate(zip(m.winners, m.servers, m.states_after, strict=True)):
                game_over = st.games != prev.games or st.sets != prev.sets or st.is_over
                set_over = st.sets != prev.sets or st.is_over
                if st.tiebreak or not game_over:
                    sc = (
                        (str(st.points[0]), str(st.points[1]))
                        if st.tiebreak
                        else (POINT_NAMES[min(st.points[0], 4)], POINT_NAMES[min(st.points[1], 4)])
                    )
                else:
                    sc = ("0", "0")
                games = prev.games if set_over else st.games
                if set_over:
                    games = (prev.games[0] + (w == 0), prev.games[1] + (w == 1))
                wr.writerow([
                    m.match_id, _hms(m.elapsed_s[i]), set_no, games[0], games[1],
                    (w + 1) if set_over else 0, game_no, (w + 1) if game_over else 0,
                    i + 1, w + 1, s + 1, "", "", sc[0], sc[1],
                    int(m.aces[i] and s == 0), int(m.aces[i] and s == 1),
                    int(m.double_faults[i] and s == 0), int(m.double_faults[i] and s == 1),
                ])  # fmt: skip
                if game_over:
                    game_no += 1
                if set_over:
                    set_no += 1
                prev = st
    with m_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(SLAM_MATCH_HEADER)
        for m in matches:
            num = int(m.match_id.rsplit("-", 1)[-1])
            wr.writerow([m.match_id, year, slam, num, m.player_a, m.player_b, "Complete",
                         m.winner + 1, "", "R32", "", "", "", "", "", ""])  # fmt: skip
    return pts_path, m_path


# ---------------------------------------------------------------------------
# Sackmann tennis_pointbypoint
# ---------------------------------------------------------------------------

PBP_HEADER = ["pbp_id", "date", "tny_name", "tour", "draw", "server1", "server2",
              "winner", "pbp", "score", "adf_flag", "wh_minutes"]  # fmt: skip


def pbp_string(m: SimulatedMatch) -> str:
    out: list[str] = []
    prev = ScoreState(server=m.first_server)
    for i, (w, s, st) in enumerate(zip(m.winners, m.servers, m.states_after, strict=True)):
        if w == s:
            out.append("A" if m.aces[i] else "S")
        else:
            out.append("D" if m.double_faults[i] else "R")
        if st.sets != prev.sets or st.is_over:
            if not st.is_over:
                out.append(".")
        elif st.games != prev.games:
            out.append(";")
        elif st.tiebreak and st.server != s:
            out.append("/")
        prev = st
    return "".join(out)


def score_string(m: SimulatedMatch) -> str:
    sets: list[str] = []
    prev = ScoreState(server=m.first_server)
    for st in m.states_after:
        if st.sets != prev.sets:
            w = 0 if st.sets[0] > prev.sets[0] else 1
            g = list(prev.games)
            g[w] += 1
            # Report from the match winner's view like Sackmann (winner first).
            sets.append(f"{g[0]}-{g[1]}")
        prev = st
    if m.winner == 1:
        sets = ["-".join(reversed(s.split("-"))) for s in sets]
    return " ".join(sets)


def write_pbp_file(
    path: Path, matches: list[SimulatedMatch], dates: list[datetime], tournament: str
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(PBP_HEADER)
        for k, (m, d) in enumerate(zip(matches, dates, strict=True)):
            # server1 must be whoever served first.
            s1, s2 = (m.player_a, m.player_b) if m.first_server == 0 else (m.player_b, m.player_a)
            winner = 1 if (m.winner == m.first_server) else 2
            wr.writerow([
                9_000_000 + k, d.strftime("%d %b %y"), tournament, "ATP", "Main", s1, s2,
                winner, pbp_string(m), score_string(m), 0, round(m.duration_s / 60),
            ])  # fmt: skip
    return path


# ---------------------------------------------------------------------------
# Betfair BASIC stream files
# ---------------------------------------------------------------------------


def write_betfair_market(
    path: Path,
    market_id: str,
    event_id: str,
    match: SimulatedMatch,
    start: datetime,
    prob_a: Callable[[ScoreState], float],
    rng: np.random.Generator,
    selection_ids: tuple[int, int],
    preplay_minutes: int = 60,
    noise_sd: float = 0.15,
) -> Path:
    """Write a BASIC-style file: one message per minute with last traded prices.

    ``prob_a(state)`` gives the "market" probability that player A wins from a score
    state. Noise is added on the logit scale and prices are rounded to ticks, so
    the implied probabilities carry a realistic small overround.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = (match.player_a, match.player_b)
    surname = lambda n: n.split()[-1]  # noqa: E731
    event_name = f"{surname(names[0])} v {surname(names[1])}"

    def definition(in_play: bool, status: str, settled: bool = False) -> dict:
        runners = []
        for k, sel in enumerate(selection_ids):
            r = {"status": "ACTIVE", "sortPriority": k + 1, "id": sel, "name": names[k]}
            if settled:
                r["status"] = "WINNER" if match.winner == k else "LOSER"
            runners.append(r)
        return {
            "bspMarket": False, "turnInPlayEnabled": True, "persistenceEnabled": True,
            "marketBaseRate": 5.0, "eventId": event_id, "eventTypeId": "2",
            "numberOfWinners": 1, "bettingType": "ODDS", "marketType": "MATCH_ODDS",
            "marketTime": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "inPlay": in_play, "status": status, "runners": runners,
            "eventName": event_name, "name": "Match Odds", "countryCode": "GB",
        }  # fmt: skip

    def prices(p: float) -> list[dict]:
        p = float(np.clip(p, 0.005, 0.995))
        logit = np.log(p / (1 - p)) + rng.normal(0, noise_sd)
        q = 1 / (1 + np.exp(-logit))
        over = 1 + abs(rng.normal(0.01, 0.005))  # traded prices imply ~1% overround
        return [{"id": selection_ids[0], "ltp": to_tick(1 / (q * over))},
                {"id": selection_ids[1], "ltp": to_tick(1 / ((1 - q) * over))}]  # fmt: skip

    ms = lambda t: int(t.timestamp() * 1000)  # noqa: E731
    msgs = []
    t0 = start - timedelta(minutes=preplay_minutes)
    initial = ScoreState(server=match.first_server)
    msgs.append({"op": "mcm", "pt": ms(t0), "mc": [
        {"id": market_id, "marketDefinition": definition(False, "OPEN"),
         "rc": prices(prob_a(initial))}]})  # fmt: skip
    for k in range(1, preplay_minutes):
        msgs.append({"op": "mcm", "pt": ms(t0 + timedelta(minutes=k)),
                     "mc": [{"id": market_id, "rc": prices(prob_a(initial))}]})  # fmt: skip
    msgs.append({"op": "mcm", "pt": ms(start), "mc": [
        {"id": market_id, "marketDefinition": definition(True, "OPEN")}]})  # fmt: skip

    ends = np.asarray(match.elapsed_s)
    minute = 1
    while minute * 60 < match.duration_s:
        idx = int(np.searchsorted(ends, minute * 60, side="right"))
        state = initial if idx == 0 else match.states_after[idx - 1]
        msgs.append({"op": "mcm", "pt": ms(start + timedelta(minutes=minute)),
                     "mc": [{"id": market_id, "rc": prices(prob_a(state))}]})  # fmt: skip
        minute += 1
    end = start + timedelta(seconds=match.duration_s)
    msgs.append({"op": "mcm", "pt": ms(end + timedelta(seconds=30)), "mc": [
        {"id": market_id, "marketDefinition": definition(True, "SUSPENDED")}]})  # fmt: skip
    closed = definition(True, "CLOSED", settled=True)
    msgs.append({"op": "mcm", "pt": ms(end + timedelta(minutes=5)), "mc": [
        {"id": market_id, "marketDefinition": closed}]})  # fmt: skip

    payload = "\n".join(json.dumps(m, separators=(",", ":")) for m in msgs) + "\n"
    path.write_bytes(bz2.compress(payload.encode()))
    return path


def heuristic_prob_a(fmt: MatchFormat, base: float = 0.5) -> Callable[[ScoreState], float]:
    """Crude market model for parser fixtures only: a logistic on sets and games lead."""

    def f(s: ScoreState) -> float:
        if s.is_over:
            return 1.0 if s.winner == 0 else 0.0
        x = (
            np.log(base / (1 - base))
            + 1.2 * (s.sets[0] - s.sets[1])
            + 0.25 * (s.games[0] - s.games[1])
        )
        return float(1 / (1 + np.exp(-x)))

    return f


UTC = UTC


# ---------------------------------------------------------------------------
# Sackmann tennis_atp
# ---------------------------------------------------------------------------

ATP_HEADER = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
    "match_num", "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age", "loser_id", "loser_seed", "loser_entry",
    "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age", "score", "best_of",
    "round", "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
]  # fmt: skip


def serve_stats(m: SimulatedMatch, player: int) -> dict:
    """Serve-point counts for one player, split into first/second serve like Sackmann."""
    served = [i for i, s in enumerate(m.servers) if s == player]
    svpt = len(served)
    won = sum(m.winners[i] == player for i in served)
    first_in = round(0.62 * svpt)
    first_won = min(round(0.72 * won), first_in)
    return dict(svpt=svpt, first_in=first_in, first_won=first_won, second_won=won - first_won,
                aces=sum(m.aces[i] for i in served),
                dfs=sum(m.double_faults[i] for i in served))  # fmt: skip


def atp_row(m: SimulatedMatch, tourney: dict, match_num: int, rnd: str) -> list:
    w, l_ = m.winner, 1 - m.winner
    names = (m.player_a, m.player_b)
    ws, ls = serve_stats(m, w), serve_stats(m, l_)
    pid = lambda name: 100000 + zlib.crc32(name.encode()) % 10**5  # noqa: E731
    return [
        tourney["id"], tourney["name"], tourney["surface"], 32, tourney["level"],
        tourney["date"].strftime("%Y%m%d"), match_num, pid(names[w]), "",
        "", names[w], "R", "", "", "", pid(names[l_]), "", "", names[l_],
        "R", "", "", "", score_string(m), m.fmt.best_of, rnd, round(m.duration_s / 60),
        ws["aces"], ws["dfs"], ws["svpt"], ws["first_in"], ws["first_won"], ws["second_won"],
        "", "", "", ls["aces"], ls["dfs"], ls["svpt"], ls["first_in"], ls["first_won"],
        ls["second_won"], "", "", "", "", "", "", "",
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Demo world: latent skills -> history, slam point-by-point and Betfair markets
# ---------------------------------------------------------------------------

_SYLLABLES = ["kar", "ven", "sol", "mir", "tad", "lun", "bor", "pex", "qui", "zan", "dov",
              "rel", "fin", "gor", "hal", "jex"]  # fmt: skip
_FIRST = ["Ari", "Bo", "Cyr", "Dax", "Eli", "Fen", "Gus", "Hal", "Ivo", "Jon", "Kai", "Leo"]
SLAMS_2023 = {  # (start date, final-set rule in 2023)
    "ausopen": datetime(2023, 1, 16), "frenchopen": datetime(2023, 5, 29),
    "wimbledon": datetime(2023, 7, 3), "usopen": datetime(2023, 8, 28),
}  # fmt: skip
TOUR_SERVE, TOUR_RETURN = 0.635, 0.365
SURFACE_SERVE_ADJ = {"Hard": 0.0, "Clay": -0.02, "Grass": 0.02}


def _surface_for_week(d: datetime) -> str:
    if d.month in (4, 5) or (d.month == 6 and d.day < 10):
        return "Clay"
    if (d.month == 6 and d.day >= 10) or (d.month == 7 and d.day < 15):
        return "Grass"
    return "Hard"


@dataclass
class DemoWorld:
    names: list[str]
    serve: np.ndarray
    ret: np.ndarray

    def p_serve(self, i: int, j: int, surface: str) -> float:
        p = self.serve[i] - (self.ret[j] - TOUR_RETURN) + SURFACE_SERVE_ADJ[surface]
        return float(np.clip(p, 0.45, 0.85))


def make_world(n_players: int, rng: np.random.Generator) -> DemoWorld:
    combos = [a + b for a in _SYLLABLES for b in _SYLLABLES if a != b]
    picks = rng.choice(len(combos), size=n_players, replace=False)
    names = [f"{_FIRST[k % len(_FIRST)]} {combos[c].capitalize()}" for k, c in enumerate(picks)]
    serve = np.clip(rng.normal(TOUR_SERVE, 0.03, n_players), 0.55, 0.75)
    ret = np.clip(rng.normal(TOUR_RETURN, 0.025, n_players), 0.28, 0.45)
    return DemoWorld(names, serve, ret)


def build_demo_dataset(
    out_dir: Path,
    prob_a_factory: Callable[[float, float, MatchFormat], Callable[[ScoreState], float]],
    n_players: int = 48,
    matches_per_slam: int = 24,
    seed: int = 2023,
    market_skill_sd: float = 0.01,
    form_sd: float = 0.015,
) -> dict:
    """Write a complete synthetic dataset in upstream formats under ``out_dir``.

    * ``atp/atp_matches_2022.csv``, ``atp/atp_matches_2023.csv``: weekly bo3 events plus
      the slams, simulated from latent skills.
    * ``slam/2023-<slam>-points.csv`` (+ matches): target matches, point by point.
    * ``betfair/<market>.bz2``: one MATCH_ODDS market per target match.

    Each target match has "form" noise (N(0, form_sd) on each serve rate) that the
    historical stats cannot see. The market sees the true rates with error
    N(0, market_skill_sd), so the market should be hard to beat. That is the honest
    null this demo is built to reproduce.
    """
    rng = np.random.default_rng(seed)
    world = make_world(n_players, rng)
    fmt_bo3 = MatchFormat(3, "tiebreak")
    atp_rows: dict[int, list] = {2022: [], 2023: []}

    def play(i, j, surface, fmt, mid, form=(0.0, 0.0)):
        pa = world.p_serve(i, j, surface) + form[0]
        pb = world.p_serve(j, i, surface) + form[1]
        return simulate_match(mid, world.names[i], world.names[j], pa, pb, fmt, rng), pa, pb

    # Weekly tour events (bo3), skipping slam fortnights.
    week = datetime(2022, 1, 3)
    slam_weeks = {d + timedelta(days=7 * k) for d in SLAMS_2023.values() for k in (0, 1)}
    slam_weeks |= {d.replace(year=2022) for d in SLAMS_2023.values()}
    t_id = 0
    while week < datetime(2023, 12, 25):
        if week not in slam_weeks:
            surface = _surface_for_week(week)
            tourney = dict(id=f"{week.year}-{t_id:04d}", name=f"Demo Open {t_id}",
                           surface=surface, level="A", date=week)  # fmt: skip
            for k in range(16):
                i, j = rng.choice(n_players, 2, replace=False)
                m, _, _ = play(i, j, surface, fmt_bo3, f"h{t_id}-{k}")
                atp_rows[week.year].append(atp_row(m, tourney, k + 1, "R32"))
            t_id += 1
        week += timedelta(days=7)

    # 2022 slams go into history only.
    for slam, start in SLAMS_2023.items():
        d = start.replace(year=2022)
        surface = _surface_for_week(d) if slam != "usopen" else "Hard"
        tourney = dict(id=f"2022-{slam}", name=slam, surface=surface, level="G", date=d)
        fmt = MatchFormat(5, "advantage" if slam in ("frenchopen",) else "super_tiebreak")
        for k in range(24):
            i, j = rng.choice(n_players, 2, replace=False)
            m, _, _ = play(i, j, surface, fmt, f"s22{slam}-{k}")
            atp_rows[2022].append(atp_row(m, tourney, k + 1, "R64"))

    # 2023 slams: the target matches.
    bf_dir = out_dir / "betfair"
    market_no = 0
    truth = []
    for slam, start in SLAMS_2023.items():
        surface = {"ausopen": "Hard", "frenchopen": "Clay", "wimbledon": "Grass",
                   "usopen": "Hard"}[slam]  # fmt: skip
        fmt = MatchFormat(5, "super_tiebreak")
        tourney = dict(id=f"2023-{slam}", name=slam, surface=surface, level="G", date=start)
        sims = []
        for k in range(matches_per_slam):
            i, j = rng.choice(n_players, 2, replace=False)
            form = tuple(rng.normal(0, form_sd, 2))
            mid = f"2023-{slam}-{1101 + k}"
            m, pa, pb = play(i, j, surface, fmt, mid, form)
            sims.append(m)
            atp_rows[2023].append(atp_row(m, tourney, k + 1, "R128"))
            day = start + timedelta(days=k // 4, hours=11 + 2 * (k % 4))
            mkt_pa = pa + rng.normal(0, market_skill_sd)
            mkt_pb = pb + rng.normal(0, market_skill_sd)
            market_no += 1
            market_id = f"1.2{market_no:08d}"
            write_betfair_market(
                bf_dir / f"{market_id}.bz2", market_id, f"3{market_no:07d}", m,
                day.replace(tzinfo=UTC), prob_a_factory(mkt_pa, mkt_pb, fmt), rng,
                (5_000_000 + 2 * market_no, 5_000_001 + 2 * market_no), preplay_minutes=30,
                noise_sd=0.08,
            )  # fmt: skip
            truth.append(dict(match_id=mid, true_pa=pa, true_pb=pb, market_pa=mkt_pa,
                              market_pb=mkt_pb, winner=m.winner))  # fmt: skip
        write_slam_files(out_dir / "slam", 2023, slam, sims)

    (out_dir / "atp").mkdir(parents=True, exist_ok=True)
    for year, rows in atp_rows.items():
        with (out_dir / "atp" / f"atp_matches_{year}.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(ATP_HEADER)
            wr.writerows(rows)
    with (out_dir / "truth.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(truth[0]))
        wr.writeheader()
        wr.writerows(truth)
    return dict(players=n_players, target_matches=len(truth),
                history_matches=sum(len(r) for r in atp_rows.values()))  # fmt: skip
