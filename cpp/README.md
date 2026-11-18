# CourtEdge C++ core

C++20 library with the latency-sensitive parts of the project:

| Header | What it does |
|---|---|
| `score.hpp` | Tennis score state machine (mirrors `courtedge.markov.score`) |
| `markov.hpp` | Exact Markov engine; tables precomputed per (pa, pb), immutable and thread-safe |
| `simulate.hpp` | Multithreaded Monte Carlo used to check the engine |
| `market.hpp` | De-vigging, Betfair tick ladder, flat and Kelly staking |
| `backtest.hpp` | Event-driven backtest loop and a thread-pool runner over many matches |

The vision pipeline (Phase 6: OpenCV + ONNX Runtime) will be added here too.

## Build

Ubuntu/Debian:

```bash
sudo apt-get install -y cmake ninja-build g++ libgtest-dev libbenchmark-dev
cmake -S cpp -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure      # run from the repo root (fixtures)
./build/courtedge_bench
```

macOS: `brew install cmake ninja googletest google-benchmark`, then the same commands.
If GoogleTest or Google Benchmark are not installed, CMake downloads them (FetchContent).

### Python extension

```bash
uv run cmake -S cpp -B build -G Ninja -DCOURTEDGE_BUILD_PYTHON=ON \
    -DPython_EXECUTABLE=$(uv run which python)
cmake --build build          # writes src/courtedge/_core*.so
```

`scripts/run_backtest.py` then uses the C++ backtest automatically (`--engine auto`).
`tests/test_core_parity.py` checks that it produces exactly the same bets as the Python one.

## Design notes

- **Engine tables, not recursion at query time.** A `MarkovEngine` fills every
  game/tiebreak/set/match table in its constructor (~13 µs). After that a query is a
  few lookups (~17 ns), and the object can be shared across threads without locks.
- **Advantage sets** are normalised once both players reach six games, so the tables
  stay bounded however long the set runs.
- **Backtest sweep.** Ticks and points are both time-sorted, so the last visible point
  is a pointer that only moves forward: O(ticks + points) per match.
- **`run_many`** hands matches to threads through an atomic counter, which keeps load
  balanced when match lengths vary a lot. Each result slot is written by one thread only.
