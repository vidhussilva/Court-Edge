"""Regenerate the small parser fixtures in tests/fixtures/.

Fixtures are synthetic but use the exact upstream file formats. Seeds are searched so
the men's match covers the awkward cases: a fifth set and at least one tiebreak.

    uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from courtedge.data import synthetic as syn
from courtedge.markov.score import FinalSet, MatchFormat

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _search(match_id, a, b, pa, pb, fmt, accept, start_seed=0):
    for seed in range(start_seed, start_seed + 5000):
        m = syn.simulate_match(match_id, a, b, pa, pb, fmt, np.random.default_rng(seed))
        if accept(m):
            return m
    raise RuntimeError(f"no seed satisfied the constraints for {match_id}")


def main() -> None:
    bo5 = MatchFormat(5, FinalSet.SUPER_TIEBREAK)
    bo3 = MatchFormat(3, FinalSet.SUPER_TIEBREAK)
    men = _search(
        "2023-wimbledon-1101", "Carlos Alcaraz", "Novak Djokovic", 0.66, 0.67, bo5,
        lambda m: sum(m.states_after[-1].sets) == 5 and any(s.tiebreak for s in m.states_after),
    )  # fmt: skip
    women = _search(
        "2023-wimbledon-2101", "Marketa Vondrousova", "Ons Jabeur", 0.56, 0.58, bo3,
        lambda m: sum(m.states_after[-1].sets) == 3,
    )  # fmt: skip
    syn.write_slam_files(FIXTURES / "slam", 2023, "wimbledon", [men, women])

    tour = MatchFormat(3, FinalSet.TIEBREAK)
    t1 = _search("t1", "Jannik Sinner", "Daniil Medvedev", 0.65, 0.64, tour,
                 lambda m: any(s.tiebreak for s in m.states_after))  # fmt: skip
    t2 = _search("t2", "Holger Rune", "Casper Ruud", 0.62, 0.63, tour, lambda m: True, 100)
    syn.write_pbp_file(
        FIXTURES / "pbp_matches_atp_main_current.csv", [t1, t2],
        [datetime(2023, 10, 1), datetime(2023, 10, 2)], "Beijing",
    )  # fmt: skip

    start = datetime(2023, 7, 16, 14, 0, tzinfo=syn.UTC)
    syn.write_betfair_market(
        FIXTURES / "betfair" / "1.216000001.bz2", "1.216000001", "32500001", men, start,
        syn.heuristic_prob_a(bo5, 0.45), np.random.default_rng(7), (2249229, 10138157),
        preplay_minutes=10,
    )  # fmt: skip
    print(f"fixtures written to {FIXTURES}")


if __name__ == "__main__":
    main()
