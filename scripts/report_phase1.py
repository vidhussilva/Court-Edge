"""Generate reports/phase1.md: engine vs Monte Carlo grid, timings, match trace.

uv run python scripts/report_phase1.py            # ~10 s with the C++ extension, ~3 min without
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from courtedge.markov import engine
from courtedge.markov.score import FinalSet, MatchFormat, ScoreState
from courtedge.markov.simulate import simulate_win_probabilities

try:
    from courtedge import _core
except ImportError:  # pure-Python fallback
    _core = None

ROOT = Path(__file__).resolve().parents[1]
P_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
PB = 0.62
N = 100_000
STATES = {
    "Bo3, 0-0": (MatchFormat(3), ScoreState()),
    "Bo3, deuce, B serving": (MatchFormat(3), ScoreState(games=(4, 5), points=(3, 3), server=1)),
    "Bo3, tiebreak 5-5": (MatchFormat(3), ScoreState(
        games=(6, 6), points=(5, 5), tiebreak=True, tiebreak_first_server=1)),
    "Bo5 super-TB, 2-2 sets": (MatchFormat(5, FinalSet.SUPER_TIEBREAK), ScoreState(sets=(2, 2))),
    "Bo5 advantage, 2-2 sets, 6-6": (
        MatchFormat(5, FinalSet.ADVANTAGE), ScoreState(sets=(2, 2), games=(6, 6), server=1)),
    "Bo5 TB at 12-12, 2-2 sets": (MatchFormat(5, FinalSet.TIEBREAK_12), ScoreState(sets=(2, 2))),
}  # fmt: skip


def simulate(state, pa, pb, fmt, seed):
    if _core is None:
        return simulate_win_probabilities(state, pa, pb, fmt, n=N, seed=seed)
    cs = _core.ScoreState(sets=state.sets, games=state.games, points=state.points,
                          server=state.server, tiebreak=state.tiebreak,
                          tiebreak_first_server=state.tiebreak_first_server)  # fmt: skip
    return _core.simulate(cs, pa, pb, _core.MatchFormat(fmt.best_of, fmt.final_set.value),
                          n=N, seed=seed)  # fmt: skip


def main() -> None:
    lines = ["| State | max abs error: game | set | match |", "|---|---|---|---|"]
    worst = 0.0
    for k, (name, (fmt, state)) in enumerate(STATES.items()):
        errs = np.zeros(3)
        for j, p in enumerate(P_GRID):
            ex = engine.win_probabilities(state, p, PB, fmt)
            mc = simulate(state, p, PB, fmt, seed=100 * k + j)
            errs = np.maximum(errs, np.abs([ex.game - mc.game, ex.set - mc.set,
                                            ex.match - mc.match]))  # fmt: skip
        worst = max(worst, errs.max())
        lines.append(f"| {name} | {errs[0]:.4f} | {errs[1]:.4f} | {errs[2]:.4f} |")

    fmt = MatchFormat(5, FinalSet.SUPER_TIEBREAK)
    s = ScoreState(sets=(1, 2), games=(4, 5), points=(2, 3))
    engine.clear_caches()
    t = time.perf_counter()
    engine.win_probabilities(s, 0.66, 0.62, fmt)
    py_cold = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(10_000):
        engine.win_probabilities(s, 0.66, 0.62, fmt)
    py_warm = (time.perf_counter() - t) / 10_000
    timing = [f"| Python, first query for a new (pa, pb) | {py_cold * 1e3:.2f} ms |",
              f"| Python, repeat query (memoised) | {py_warm * 1e6:.1f} µs |"]  # fmt: skip
    if _core is not None:
        cfmt = _core.MatchFormat(5, "super_tiebreak")
        cs = _core.ScoreState(sets=(1, 2), games=(4, 5), points=(2, 3))
        t = time.perf_counter()
        for _ in range(1000):
            _core.MarkovEngine(0.66, 0.62, cfmt)
        c_build = (time.perf_counter() - t) / 1000
        e = _core.MarkovEngine(0.66, 0.62, cfmt)
        t = time.perf_counter()
        for _ in range(10_000):
            e.win_probabilities(cs)
        c_query = (time.perf_counter() - t) / 10_000
        timing += [
            f"| C++ engine construction via Python | {c_build * 1e6:.1f} µs |",
            f"| C++ query via Python (includes binding overhead) | {c_query * 1e6:.2f} µs |",
        ]

    report = f"""# Phase 1: Markov win-probability engine

Points are modelled as independent given the server: `pa` = P(A wins a point on serve),
`pb` = P(B wins a point on serve). The engine (`courtedge.markov.engine`, ported to C++ in
`cpp/src/markov.cpp`) returns P(A wins) the current game, set and match from any score.
It covers best-of-3/5 and all four deciding-set rules (7-point tiebreak at 6-6, advantage,
tiebreak at 12-12, 10-point tiebreak at 6-6).

Game closed form: `P_game(0.6) = {engine.p_game(0.6):.4f}` (spec check: about 0.736).

## Engine vs Monte Carlo

{N:,} simulations per cell, `pa` in {P_GRID}, `pb = {PB}`. Tolerance 0.005.
Simulator: {"C++ (`courtedge._core.simulate`)" if _core else "NumPy (`courtedge.markov.simulate`)"}.

{chr(10).join(lines)}

Worst error over the grid: **{worst:.4f}** ({"PASS" if worst < 0.005 else "FAIL"}).
One Monte Carlo standard error at p = 0.5 is 0.0016.

Other checks in `tests/test_markov.py` and `cpp/tests/test_markov.cpp`:

- **Martingale identity** at every state of a replayed match and along long random walks:
  P(now) = q·P(after A wins point) + (1-q)·P(after B wins point), to 1e-12. The next
  states come from the separate score state machine, so this ties the engine to the
  scoring rules exactly.
- **Python/C++ parity** on 5,593 random states, to 1e-10.
- Symmetry (swap players) and equal players giving 0.5 under every format.

## Speed

| Operation | Time |
|---|---|
{chr(10).join(timing)}

C++ benchmarks: `./build/courtedge_bench` (Google Benchmark).

## Match trace

![trace](figures/phase1_match_trace_synthetic.png)

The trace above uses the **synthetic** fixture match. To plot a real match after Phase 0:
`uv run python scripts/plot_match_trace.py --match-id <id>`.
"""
    (ROOT / "reports" / "phase1.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
