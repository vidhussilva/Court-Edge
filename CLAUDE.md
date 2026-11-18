# CourtEdge
Research project: does video of a tennis match predict outcomes better than Betfair odds,
and at what horizon does any edge survive broadcast delay? Full spec in SPEC.md.
Read the relevant phase in SPEC.md before starting any work.

## Rules
- Simulation only. Never write code that places real bets or connects to a live betting account.
- Python 3.11 managed with uv. C++20 with CMake. Tests: pytest and GoogleTest.
- No look-ahead: anything used at time t may only depend on data available at time t.
  Add a test for this whenever you touch features or the backtester.
- Raw data and video live in data/raw/ (git-ignored). Never commit footage or Betfair files.
- A phase is done when tests pass, docs/phase-log.md has an entry, and results are saved in reports/.
- Prefer small, readable modules over clever abstractions. Comment non-obvious choices.
- Before running anything slow (training, full-match video processing), state the expected runtime.
