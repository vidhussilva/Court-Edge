"""Write the SYNTHETIC demo dataset to data/raw/demo/ (about 20 s).

    uv run python scripts/make_demo_data.py

Everything downstream (catalogue, backtest, reports) runs on it before real
Betfair data is available. Reports built from it say SYNTHETIC in their titles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from courtedge.data.synthetic import build_demo_dataset
from courtedge.markov.engine import win_probabilities


def engine_market(pa: float, pb: float, fmt):
    """The demo market prices every state with the Markov engine at its own skill estimates."""
    return lambda state: win_probabilities(state, pa, pb, fmt).match


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/demo")
    ap.add_argument("--matches-per-slam", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2023)
    args = ap.parse_args(argv)
    info = build_demo_dataset(Path(args.out), engine_market,
                              matches_per_slam=args.matches_per_slam, seed=args.seed)  # fmt: skip
    print(f"SYNTHETIC demo data -> {args.out}: {info}")


if __name__ == "__main__":
    main()
