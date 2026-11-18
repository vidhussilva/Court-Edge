"""Write Python engine outputs for random score states to a CSV fixture.

The C++ tests read it (cpp/tests/test_markov.cpp) and must agree to 1e-10, so the
two implementations cannot drift apart.

    uv run python scripts/export_markov_reference.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from courtedge.markov.engine import win_probabilities
from courtedge.markov.score import FinalSet, MatchFormat, ScoreState, advance

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "markov_reference.csv"
HEADER = ["pa", "pb", "best_of", "final_set", "sets_a", "sets_b", "games_a", "games_b",
          "points_a", "points_b", "server", "tiebreak", "tb_first_server",
          "game", "set", "match"]  # fmt: skip


def main(n_walks: int = 6, steps: int = 220) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for best_of in (3, 5):
        for rule in FinalSet:
            fmt = MatchFormat(best_of, rule)
            for _ in range(n_walks):
                pa, pb = rng.uniform(0.5, 0.78, size=2).round(4)
                # Start some walks at the decider so long final sets are covered.
                last = fmt.sets_to_win - 1
                state = ScoreState(sets=(last, last)) if rng.random() < 0.5 else ScoreState()
                for _ in range(steps):
                    if state.is_over:
                        break
                    w = win_probabilities(state, pa, pb, fmt)
                    rows.append([pa, pb, best_of, rule.value, *state.sets, *state.games,
                                 *state.points, state.server, int(state.tiebreak),
                                 state.tiebreak_first_server, repr(w.game), repr(w.set),
                                 repr(w.match)])  # fmt: skip
                    q = pa if state.server == 0 else 1 - pb
                    state = advance(state, 0 if rng.random() < q else 1, fmt)
    with OUT.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        wr.writerows(rows)
    print(f"{len(rows)} reference states -> {OUT}")


if __name__ == "__main__":
    main()
