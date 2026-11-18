# CourtEdge

[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Does what a camera sees in a tennis match predict outcomes better than the betting
market, and does any edge survive broadcast delay?**

CourtEdge combines an exact Markov win-probability engine, an event-driven backtester
against Betfair Exchange historical odds, and (in later phases) a computer-vision
pipeline that turns broadcast video into per-point player features. It is
**simulation only**: nothing connects to a live betting account.

## Status

| Version | Phases | State |
|---|---|---|
| V1 | 0-2: data, Markov engine, odds-only backtest | Code complete and tested. Real-data run pending Betfair download |
| V2 | 3-5: vision features, alignment, point-win model | Not started |
| V3 | 6-8: real-time C++ vision pipeline, edge vs delay, write-up | Not started (C++ core in place) |

Reports in `reports/` are currently built from a **synthetic** demo world (clearly
labelled) so the whole pipeline runs end to end. They will be regenerated from real data.

## Architecture

```
Sackmann point-by-point ─▶ score state machine ─▶ Markov engine ─▶ P(win game/set/match)
Sackmann ATP stats ─────▶ pre-match serve probabilities ──┘                │
Betfair BASIC stream ───▶ de-vigged market probability ───────────▶ backtester (delay,
                                                                    commission, staking)
                                                                           │
                                                      calibration, Brier vs market, P&L
```

| Component | Python | C++20 |
|---|---|---|
| Score state machine (tiebreaks, bo3/bo5, 4 final-set rules) | `markov/score.py` | `score.hpp` |
| Exact Markov engine | `markov/engine.py` | `markov.hpp` |
| Monte Carlo check | `markov/simulate.py` (NumPy) | `simulate.hpp` (multithreaded) |
| Backtester | `market/backtest.py` | `backtest.hpp` (thread pool) |
| Parsers, catalogue, ATP serve model, reports | `data/`, `report/` | |

The two implementations are tied together by parity tests: the engines agree on 5,593
random states to 1e-10, and both backtesters produce the same bets.

## Correctness checks

- Engine vs 100k-simulation Monte Carlo over the spec grid: worst error 0.0040 (tolerance 0.005).
- Martingale identity at every score state: P(now) = q·P(A wins point) + (1-q)·P(B wins point), to 1e-12.
- **No look-ahead**: everything after time t is perturbed (prices, scores, even the
  winner), and the test asserts that no decision at or before t changes (Python and C++).
- Pre-match serve estimates only use tournaments that must have *finished* before the
  match. Sackmann dates every match with its tournament's start date.
- P&L settled the way Betfair does it: commission on net market winnings, at each
  market's recorded rate.

## Performance (C++ core, 4-core container)

| Operation | Time |
|---|---|
| Build engine tables for one (pa, pb) | 13 µs |
| One win-probability query | 17 ns |
| 100k Monte Carlo matches, 1 → 4 threads | 634 → 176 ms |
| Backtest 200 best-of-5 matches, 1 → 4 threads | 6.2 → 1.5 ms |

## Run it

```bash
uv sync --all-groups
just test        # pytest + ctest (builds the C++ core and Python extension)
just demo        # whole pipeline on the synthetic demo world
just bench       # Google Benchmark
```

Real data: `scripts/download_data.sh` fetches the Sackmann repos; Betfair BASIC tennis
files need a (free) account at historicdata.betfair.com and go in `data/raw/betfair/`.
Then `just data markov backtest`. See [`SPEC.md`](SPEC.md) for the full plan and
[`cpp/README.md`](cpp/README.md) for C++ build details.

## Data credits

Point-by-point and match data: Jeff Sackmann, `tennis_slam_pointbypoint`,
`tennis_pointbypoint`, `tennis_atp` (CC BY-NC-SA 4.0). Odds: Betfair Historical Data.
No footage or Betfair files are committed.
