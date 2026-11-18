#!/usr/bin/env bash
# Fetch the public Sackmann datasets into data/raw/ (git-ignored).
# Betfair BASIC files need an account: download them from
# https://historicdata.betfair.com (Tennis, BASIC, one month around a Grand Slam)
# and put the .tar files in data/raw/betfair/.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
for repo in tennis_slam_pointbypoint tennis_pointbypoint tennis_atp; do
  if [ -d "$repo" ]; then
    git -C "$repo" pull --ff-only
  else
    git clone --depth 1 "https://github.com/JeffSackmann/$repo.git"
  fi
done
mkdir -p betfair
echo "Sackmann data ready in data/raw/. Put Betfair BASIC .tar files in data/raw/betfair/."
