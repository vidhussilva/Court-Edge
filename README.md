# CourtEdge

[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Can what a camera sees in a tennis match predict the result better than the betting
market? And if there is an edge, does it survive the delay between the real point and
the broadcast?**

The end goal is one chart: simulated edge against broadcast delay, with a line for each
forecast horizon (next game, next set, match). Getting there needs three parts. A
probability model of tennis scoring, a backtester that replays Betfair Exchange prices
honestly, and a vision pipeline that turns broadcast footage into per-point
measurements of how each player is moving.

This is a research project and runs in simulation only. Nothing here connects to a
betting account or places a bet.

---

## Where it stands

| Version | Phases | Status | Target |
|---|---|---|---|
| **V1** | 0-2: data, Markov engine, odds-only backtest | Code done, 171 Python + 27 C++ tests passing in CI. Waiting on a Betfair data download for the real-data run | 1 Dec 2026 |
| **V2** | 3-5: vision features, video/odds alignment, point-win model | Not started | 10 Jan 2027 |
| **V3** | 6-8: real-time C++ vision pipeline, edge-vs-delay experiment, write-up | C++ core in place; vision part not started | 14 Feb 2027 |

The reports in [`reports/`](reports/) were produced from a **synthetic** dataset I
generate in the exact file formats of the real sources (see
[Results so far](#results-so-far)). Every synthetic report says so in its title. They
prove the pipeline runs end to end; they are not findings about tennis. The real-data
run is the next step.

---

## Why this question

Betfair's tennis markets are liquid and quick to react, so beating them on information
everyone has (the score, the players' records) is unlikely. A camera sees things the
scoreboard doesn't: how far a player is running, whether their top speed is dropping,
how long they take between points, how deep behind the baseline they're being pushed.
If any of that predicts what happens next, the interesting question is how long the
signal lasts. TV pictures run several seconds to minutes behind the court, so a
"live" feature is already stale. My guess is that any edge, if one exists, sits at the
longer horizons (the next set rather than the next point). The project is set up to
measure that, and a careful "no edge" is a perfectly good answer.

---

## How it works

```
Sackmann point-by-point ──▶ score state machine ──▶ Markov engine ──▶ P(win game / set / match)
Sackmann ATP match stats ─▶ pre-match serve-point probabilities ─┘               │
Betfair historical stream ─▶ de-vigged market probability ──────────────▶ backtester
                                                                (delay, commission, staking)
Broadcast video (V2) ─────▶ court homography, player tracking, pose ─▶ per-point features
                                                                                 │
                                                  calibration, Brier vs market, P&L, edge vs delay
```

**Score state machine** (`markov/score.py`, `cpp/src/score.cpp`). One definition of
tennis scoring that everything else uses: tiebreak serve rotation, who serves after a
tiebreak, best-of-3 and best-of-5, and the four deciding-set rules the Grand Slams have
used since 2011 (advantage, tiebreak at 6-6, tiebreak at 12-12, 10-point tiebreak).

**Markov engine** (`markov/engine.py`, `cpp/src/markov.cpp`). If each player wins a
point on serve with a fixed probability, the chance of winning the current game, set
and match follows exactly from the score. The game has a closed form
(P_game(0.6) = 0.7357). Sets and matches are memoised recursions over the score.
Deuce, a level tiebreak and a long advantage set would recurse forever, so each one is
solved as a small linear equation instead.

**Pre-match serve model** (`data/atp.py`). Each player's serve and return
points-won rates over the last 52 weeks. These are shrunk toward the player's overall
rate when they've played little on that surface, then adjusted for the opponent's
return strength (Barnett and Clarke, 2005).

**Backtester** (`market/backtest.py`, `cpp/src/backtest.cpp`). Steps through the market
one minute at a time. At each tick the model only sees the score as it was `delay`
seconds earlier, the market price is de-vigged, and a bet is placed when the two
disagree by more than a threshold. Bets get one tick of slippage. Settlement works the
way Betfair does it: commission on net market winnings, at the rate recorded in each
market's own file.

**C++ core** (`cpp/`). The engine, a Monte Carlo simulator and the backtest loop,
ported to C++20 for the parts that will need speed: the Phase 7 grid runs every match
many times over. Python calls it through pybind11, and parity tests check the two
versions give the same bets. Phase 6 adds the real-time vision pipeline here.

---

## Decisions and things that went wrong

These are the parts I'd want to talk about, because each one changed the code.

**A look-ahead leak in the serve stats.** Sackmann dates every match with its
tournament's *start* date, so a final played on day 13 looks like it happened on day 1.
Filtering on `date < match_date` would let later rounds of the same event leak into a
first-round estimate. The rule now is that a match only counts if its tournament must
have *finished* before the target match: start date plus 15 days for slams and 8 for
everything else. It throws away a little recent information. That's the right trade
here, and `tests/test_atp.py` checks it.

**Normalising the score lost information.** I map deuce-like scores to a canonical
form (5-5 in points becomes 3-3) so the memo tables stay small. In a tiebreak that
erases who served first, and that decides who serves the next set. The state now
carries the tiebreak's first server explicitly.

**A Monte Carlo test failed at 3.8 standard errors.** The C++ engine-vs-simulation grid
failed one cell out of 36. Before touching the tolerance I re-ran that cell across 8
seeds: the mean was 0.5009 against an exact 0.5000, so there was no bias, just an
unlucky draw from running many comparisons at about 3σ each. The C++ test now uses
400k simulations, which puts the 0.005 tolerance at about 6σ.

**The look-ahead test found a different bug.** The no-look-ahead test scrambles
everything after time t (prices, scores, even the winner) and asserts that no earlier
decision changes. My first version scrambled score columns independently, which
produced impossible scores like 7-7 in a normal set. The engine recursed on them until
Python's stack ran out. I fixed the test to shuffle whole states, and the engine now
rejects impossible scores up front in both languages. In C++ that also keeps the table
lookups in bounds.

**Report generation went from 37 s to 4 s.** Profiling showed 29 s inside the
bootstrap, which rebuilt a DataFrame with `pd.concat` for each of 1,000 resamples. Every
statistic I report is a ratio of sums (Brier is mean squared error, ROI is net over
staked), so I collapse to per-match sums once and resample those with NumPy. The
intervals came out identical, since the random draws line up. That speed matters for
Phase 7, which repeats this for every cell of the grid.

**Resampling whole matches.** One match produces hundreds of in-play forecasts all
scored against the same outcome. Bootstrapping individual ticks would give confidence
intervals that look far tighter than they really are, so every interval in the reports
resamples whole matches.

**What commission to charge.** Published figures for Betfair's tennis commission
disagree, and most come from affiliate sites. The historical files record a
`marketBaseRate` for each market, so the backtester uses that and falls back to 5% only
when it's missing.

**CI caught a file I never committed.** My `.gitignore` excludes `*.bz2` (Betfair
files) with an exception for test fixtures, but the exception didn't match
subfolders. The Betfair fixture existed on my machine and nowhere else, so tests passed
locally and failed in CI. The fix was one line in `.gitignore`. It's also a good
argument for having CI from the first commit, because nothing on my machine would have
caught it.

---

## How I check it's right

- **Engine vs simulation.** 100k simulated matches per cell, serve probabilities 0.50 to
  0.75, six score states covering every deciding-set rule. Worst error 0.0040 against
  a tolerance of 0.005 ([`reports/phase1.md`](reports/phase1.md)).
- **Martingale identity.** For every score state in a replayed match,
  P(now) = q·P(after A wins the point) + (1-q)·P(after B wins it), to 1e-12. The next
  states come from the separate score state machine, so this ties the engine to the
  scoring rules exactly.
- **Python/C++ parity.** The two engines agree on 5,593 random states to 1e-10. The two
  backtesters produce identical bets under three different configurations.
- **No look-ahead.** Perturbation tests in both languages, plus a test that the
  delay hides exactly the points it should.
- **Hand-calculated P&L.** For example, two bets in one market that net to zero pay
  no commission.
- **Parser self-checks.** The point-by-point loader rebuilds the score itself and
  records any disagreement with the source file's server or winner columns, instead of
  silently dropping the match.

---

## Performance

C++ core on a 4-core machine, measured with Google Benchmark (`just bench`):

| Operation | Time |
|---|---|
| Build engine tables for one pair of serve probabilities | 13 µs |
| One win-probability query | 17 ns |
| 100k Monte Carlo matches, 1 → 4 threads | 634 → 176 ms |
| Backtest 200 best-of-5 matches, 1 → 4 threads | 6.2 → 1.5 ms |

The engine builds every table when it's constructed, so after that it's read-only and
threads can share it without locks. The backtest finds the last visible point with a
pointer that only moves forward, since ticks and points are both in time order. That
makes each match one linear pass. Matches go to threads through an atomic counter,
which keeps the load balanced when one five-setter is three times longer than a
straight-sets match.

---

## Results so far

**Synthetic only.** I built a small simulated tour with hidden player skill levels,
two seasons of match history, four Grand Slams of point-by-point data and a Betfair
market for each match. The market is deliberately better informed than my model: it
sees each match's true form, and the model only has historical stats. So the model
*should* lose, and the useful check is whether the pipeline reports that honestly.

| Forecast (in-play, P(match win)) | Brier [95% CI] |
|---|---|
| Market | 0.156 [0.120, 0.194] |
| Markov model, 60 s delay | 0.178 [0.137, 0.219] |
| Model minus market | +0.022 [-0.000, 0.046] |

The strategy placed 813 bets across 87 matches for an ROI of +0.8%, with a 95% interval
of [-32%, +38%]. That is noise, which is the expected answer. Full details, calibration
plot and P&L curve: [`reports/phase2.md`](reports/phase2.md).

---

## What's next

**Finish V1 on real data (next).** Download a month of Betfair BASIC tennis data around a
Grand Slam, link it to Sackmann's point-by-point files, and rerun Phases 0-2. The
questions I want answered first:

- How well calibrated is the market in-play compared with pre-match?
- How much worse does the stats-only model get as the delay grows?
- How many matches can the catalogue link with confidence? Name matching across
  sources is messier than in the synthetic data.

**V2: vision (over the Christmas break).**
- *Court geometry.* Detect court line intersections and fit a homography from pixels to
  metres. Detect camera cuts and only keep frames from the main broadcast angle. Target:
  under 5 px reprojection error.
- *Players.* YOLO plus ByteTrack, keep the two tracks that are on or near the court,
  and label them near and far.
- *Pose.* YOLO pose keypoints for both players, smoothed over time.
- *Ball.* A TrackNet-style heatmap model. Optional for V2, since rally length can come
  from shot detection.
- *Point segmentation.* Detect serve starts and point ends, aiming to find at least
  90% of points on a hand-checked match.
- *Features per player per point:* distance run, mean and peak speed, peak speed over
  the last 10 points relative to the first set (the main fatigue signal), recovery time
  between points, depth behind the baseline, rally length, and the variance of the
  serve motion.
- *Alignment.* Match the points found in the video to the point-by-point rows by
  sequence (dynamic programming over server order and game boundaries), and anchor
  video time to the moment the market goes in-play.
- *Point-win model.* Logistic regression on the score alone, then score plus vision,
  then gradient boosting, then a small GRU over the point sequence. Split by match and
  ordered by date. The ablation with and without vision is the result that matters,
  and I'll report it even if vision adds nothing.

**V3: real time and the main result.**
- Export the detection, pose and ball models to ONNX. Run them in a C++ pipeline with
  one thread per stage and bounded queues between stages: decode, preprocess,
  inference, postprocess, tracking and homography, features. Target 25 fps, with p50
  and p95 latency per stage and a log of every optimisation tried and what it did.
- Parity tests between the C++ and Python features, the same way the engine is tested
  now.
- The edge-vs-delay grid: delays of 60 s, 2 min and 5 min, horizons of next game, next
  set and match, comparing the market, the score-only model and score plus vision.
  This is the chart the project exists to produce.

**Open questions I'm still thinking about:**
- Free Betfair data is one-minute resolution, so delays under a minute can't be tested.
  If the 60 s results are interesting, it would be worth buying the 1-second data for a
  handful of matches.
- Matches with full footage available skew toward finals and big names, which are also
  the most liquid markets. I'll compare odds-only results on all matches against the
  video subset to see how much that biases things.
- Until Phase 4, point times are either Sackmann's `ElapsedTime` anchored to the
  in-play minute, or estimated by spreading the points between in-play start and market
  suspension. The second method uses the match's end time, which a live trader wouldn't
  know. Video alignment replaces both.

---

## Limitations (current)

- Betfair BASIC data has last-traded prices only: no volume, no order book. Assuming a
  fill at one tick worse than the last trade is optimistic for larger stakes.
- The serve model assumes points are independent given the server. That's the
  standard assumption and a reasonable baseline, but momentum and pressure points are
  exactly what the point-win model in Phase 5 is meant to test.
- The strategy settings in `configs/backtest.yaml` were fixed before running anything.
  If I try other settings, every one goes in the report, not just the best.

---

## Running it

```bash
uv sync --all-groups      # Python 3.11 environment
just test                 # builds the C++ core + Python extension, runs pytest and ctest
just demo                 # whole pipeline on the synthetic dataset (about 15 s)
just bench                # C++ benchmarks
```

With real data:

```bash
scripts/download_data.sh  # Sackmann repos into data/raw/
# Betfair BASIC tennis files (free account at historicdata.betfair.com) into data/raw/betfair/
just data backtest   # then: just markov
```

No GoogleTest or Google Benchmark installed? CMake fetches them. See
[`cpp/README.md`](cpp/README.md) for build notes and [`SPEC.md`](SPEC.md) for the full
plan, phase by phase.

## Layout

```
src/courtedge/
  data/      parsers (Sackmann, Betfair), catalogue, point timing, ATP serve model, synthetic data
  markov/    score state machine, exact engine, Monte Carlo check
  market/    de-vig, tick ladder, staking, backtester, C++ adapter
  report/    metrics with match-level bootstrap, plot style
cpp/         C++20 core: engine, simulator, backtester, tests, benchmarks, pybind11 bindings
scripts/     entry points that build data and write reports
reports/     generated results (phase0-2 so far)
docs/        phase log and learnings
```

## Data and credits

Point-by-point and match data from Jeff Sackmann's `tennis_slam_pointbypoint`,
`tennis_pointbypoint` and `tennis_atp` repositories (CC BY-NC-SA 4.0). Odds from
Betfair Historical Data. No match footage or Betfair files are committed; the vision
work will publish derived features and plots only.
