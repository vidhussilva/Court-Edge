# One command per phase. `just --list` shows them all.
# Real data goes in data/raw/ (see scripts/download_data.sh); `just demo` runs everything
# on the SYNTHETIC demo world instead.

# Override on the command line, e.g. just slam=data/raw/tennis_slam_pointbypoint/2024-usopen-points.csv data
slam := "data/raw/tennis_slam_pointbypoint/2023-wimbledon-points.csv"
atp := "data/raw/tennis_atp/atp_matches_2022.csv data/raw/tennis_atp/atp_matches_2023.csv"

# Python env + C++ core (tests, benchmarks, Python extension)
setup:
    uv sync --all-groups
    uv run cmake -S cpp -B build -G Ninja -DCOURTEDGE_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(uv run which python)
    cmake --build build

test: setup
    uv run pytest -q
    ctest --test-dir build --output-on-failure

bench: setup
    ./build/courtedge_bench

# Phase 0: parse sources, link them, plot one market
data:
    uv run python scripts/build_catalogue.py --slam {{slam}} --betfair data/raw/betfair --out data/interim --catalogue data/catalogue.csv
    uv run python scripts/plot_market_trace.py

# Phase 1: engine-vs-simulation report (needs no data)
markov:
    uv run python scripts/report_phase1.py

# Run `just data` first; the match trace uses the serve estimates the backtest writes.
# Phase 2: odds-only backtest, report and one match trace
backtest:
    uv run python scripts/run_backtest.py --config configs/backtest.yaml --atp {{atp}} --report reports/phase2.md
    uv run python scripts/plot_match_trace.py

# Phases 0-2 on the SYNTHETIC demo world (about 15 s)
demo:
    uv run python scripts/make_demo_data.py
    uv run python scripts/build_catalogue.py --slam data/raw/demo/slam/2023-*-points.csv --betfair data/raw/demo/betfair --out data/interim/demo --catalogue data/interim/demo/catalogue.csv
    uv run python scripts/run_backtest.py --interim data/interim/demo --catalogue data/interim/demo/catalogue.csv --atp data/raw/demo/atp/atp_matches_202*.csv --report reports/phase2.md --synthetic --truth data/raw/demo/truth.csv

fixtures:
    uv run python scripts/make_fixtures.py
    uv run python scripts/export_markov_reference.py
