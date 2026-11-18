# Phase log

One entry per completed phase: what was built, how it was verified, open issues.

## Phase 0: setup and data audit (code complete)

- Built: uv project, CMake/GoogleTest skeleton, CI; score state machine; Sackmann slam and
  tour parsers; Betfair BASIC parser; catalogue builder; point timing; market trace plot.
- Verified: 50 pytest tests on fixtures in the exact upstream formats; parser
  self-consistency (server and winner checks) reports zero issues on fixtures.
- Open: the real-data audit needs the Sackmann clones and a Betfair BASIC download (account
  required). `reports/phase0.md` lists the commands.

## Phase 1: Markov engine

- Built: exact engine (game closed form, tiebreak and advantage loops solved algebraically,
  memoised recursion for sets and matches); NumPy Monte Carlo simulator; C++20 port of the
  score machine, engine and a multithreaded simulator; pybind11 bindings; match trace plot.
- Verified: engine vs 100k-sim Monte Carlo over the spec grid, worst error 0.0040
  (`reports/phase1.md`); martingale identity to 1e-12; Python/C++ parity on 5,593 states
  to 1e-10.
- Note: the first C++ grid run failed one cell at 3.8 sigma with 100k sims. Checking
  across seeds showed no bias, so the C++ test now uses 400k sims (tolerance at about 6 sigma).
