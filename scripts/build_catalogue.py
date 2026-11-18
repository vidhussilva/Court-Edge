"""Parse raw sources and link matches across them.

    uv run python scripts/build_catalogue.py \
        --slam data/raw/tennis_slam_pointbypoint/2023-wimbledon-points.csv \
        --betfair data/raw/betfair/ --out data/interim --catalogue data/catalogue.csv

Writes points.parquet, matches.parquet, betfair.parquet and unmatched.csv to --out,
and the catalogue CSV. video_path / video_offset_s are left empty to fill in by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from courtedge.data import betfair, pbp
from courtedge.data.catalogue import build_catalogue, market_summary


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slam", nargs="*", default=[], help="<year>-<slam>-points.csv files")
    ap.add_argument("--pbp", nargs="*", default=[], help="tennis_pointbypoint CSV files")
    ap.add_argument("--betfair", required=True, help="BASIC file, directory or .tar")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--catalogue", default="data/catalogue.csv")
    ap.add_argument("--min-confidence", type=float, default=0.8)
    args = ap.parse_args(argv)

    frames = [pbp.load_slam(p) for p in args.slam] + [pbp.load_pointbypoint(p) for p in args.pbp]
    if not frames:
        ap.error("give at least one --slam or --pbp file")
    points = pd.concat([f[0] for f in frames], ignore_index=True)
    matches = pd.concat([f[1] for f in frames], ignore_index=True)
    bf = betfair.parse_path(args.betfair)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    points.to_parquet(out / "points.parquet", index=False)
    matches.to_parquet(out / "matches.parquet", index=False)
    bf.to_parquet(out / "betfair.parquet", index=False)

    catalogue, unmatched = build_catalogue(matches, market_summary(bf), args.min_confidence)
    Path(args.catalogue).parent.mkdir(parents=True, exist_ok=True)
    catalogue.to_csv(args.catalogue, index=False)
    unmatched[["match_id", "player_a", "player_b", "tournament", "year"]].to_csv(
        out / "unmatched.csv", index=False
    )
    issues = matches["parse_issues"].fillna("").ne("").sum()
    print(
        f"{len(matches)} matches parsed ({issues} with parse issues), "
        f"{bf['market_id'].nunique()} markets, {len(catalogue)} linked -> {args.catalogue}"
    )
    return catalogue


if __name__ == "__main__":
    main()
