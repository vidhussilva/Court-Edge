# Phase log

One entry per completed phase: what was built, how it was verified, open issues.

## Phase 0: setup and data audit (code complete)

- Built: uv project, CMake/GoogleTest skeleton, CI; score state machine; Sackmann slam and
  tour parsers; Betfair BASIC parser; catalogue builder; point timing; market trace plot.
- Verified: 50 pytest tests on fixtures in the exact upstream formats; parser
  self-consistency (server and winner checks) reports zero issues on fixtures.
- Open: the real-data audit needs the Sackmann clones and a Betfair BASIC download (account
  required). `reports/phase0.md` lists the commands.
