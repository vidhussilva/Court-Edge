# CourtEdge

**Does what a camera sees in a tennis match predict outcomes better than the betting market — and does any edge survive broadcast delay?**

CourtEdge is a research project: a Markov win-probability engine, a Betfair Exchange
backtester and (later) a computer-vision pipeline that turns broadcast video into
per-point player features. It is **simulation only** — nothing connects to a live
betting account.

Status: work in progress. See [`SPEC.md`](SPEC.md) for the full plan.

## Layout

```
src/courtedge/   Python package (data parsers, Markov engine, backtester, reports)
cpp/             C++20 core (engine, Monte Carlo, backtest loop), GoogleTest, benchmarks
scripts/         Entry points that build data and write reports
reports/         Generated results
```

## Setup

```bash
uv sync --all-groups          # Python 3.11 env
uv run pytest -q              # Python tests
cmake -S cpp -B build -G Ninja && cmake --build build && ctest --test-dir build
```
