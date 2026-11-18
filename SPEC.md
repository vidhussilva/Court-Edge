# CourtEdge: Vision-Based Tennis Win Probability & Market Backtester

## Overview

CourtEdge tests one question: does what a camera sees in a tennis match predict
outcomes better than the betting market already does, and at what time horizon
does any edge survive a realistic broadcast delay? The headline output is one
chart: simulated edge against delay, with one line per forecast horizon (next
game, next set, match).

The project is simulation only. No real-money trading at any stage.

```
Match video ──▶ Vision pipeline (C++ / ONNX Runtime) ──▶ per-point features ─┐
                                                                             │
Point-by-point data ──▶ score state ──▶ Point-win model ──▶ Markov engine ──▶ P(win game / set / match)
                                                                             │
Betfair historical odds ──▶ de-vigged implied probabilities ──────────────▶ Backtest simulator
                                                                    (delay, commission, staking)
                                                                             │
                                                                             ▼
                                                        Calibration + edge-vs-delay report
```

| Requirement | Where the project proves it |
|---|---|
| Proven C++ programming and problem-solving | Phase 6: multithreaded real-time inference pipeline, benchmarked |
| ML fundamentals, PyTorch | Phase 5: point-win model with baselines, ablations and calibration |
| ML frameworks and deployment | Phase 6: PyTorch → ONNX → ONNX Runtime C++, with output-parity tests |
| Practical ML/DL on vision | Phase 3: detection, pose, ball tracking, court homography |
| Written communication | Phase 8: README, write-up, results report |
| Probability and quant interest | Phases 1, 2 and 7: Markov model, de-vigging, backtesting |

## Rules

- Simulation only. Never write code that places real bets or connects to a live betting account.
- Python 3.11 managed with uv. C++20 with CMake. Tests: pytest and GoogleTest.
- No look-ahead: anything used at time t may only depend on data available at time t.
  Add a test for this whenever features or the backtester change.
- Raw data and video live in `data/raw/` (git-ignored). Never commit footage or Betfair files.
- A phase is done when tests pass, `docs/phase-log.md` has an entry, and results are saved in `reports/`.
- Prefer small, readable modules over clever abstractions. Comment non-obvious choices.

## Tech stack

Python does the prototyping and research. C++ does the real-time and hot paths.

| Area | Choice | Notes |
|---|---|---|
| Environment | Python 3.11, uv | One lockfile, reproducible |
| Data | polars / pandas, parquet via pyarrow | Every intermediate table saved as parquet |
| Betfair parsing | Betfair historic stream JSON (bz2) | BASIC tier: last traded price at 1-minute intervals |
| Vision prototyping | Ultralytics YOLO (detect + pose + ByteTrack), OpenCV | AGPL licence: fine for an open-source repo |
| Ball tracking | TrackNet-style heatmap network in PyTorch | Pretrained weights if available |
| Modelling | scikit-learn (baselines, calibration), PyTorch | Logistic regression first, neural models second |
| Export | ONNX, onnxruntime | Same models in Python and C++ |
| C++ | C++20, CMake, OpenCV, ONNX Runtime C++ API | GoogleTest for tests, Google Benchmark for timings |
| Reporting | matplotlib, scripts that write to `reports/` | Notebooks for exploring only |
| CI | GitHub Actions | Runs pytest and ctest on every push |

## Data sources

| Data | Source | What you get | Constraint |
|---|---|---|---|
| Point-by-point | `JeffSackmann/tennis_pointbypoint`, `tennis_slam_pointbypoint` | Every point of tens of thousands of pro matches | CC BY-NC-SA 4.0, no timestamps per point |
| Match results and serve stats | `JeffSackmann/tennis_atp`, `tennis_wta` | Results, rankings, per-match serve stats | Same licence |
| Odds | Betfair Historical Data | Basic tier (free): last traded price at 1-minute intervals | Account required to download |
| Broadcast video | Full-match footage you can lawfully access | Frames for the vision pipeline | Keep locally, never redistribute |
| Ball-tracking training data | TrackNet dataset | Labelled ball positions | Research use |

**Day-one risk: alignment.** Point-by-point data has no timestamps and Betfair data
has no score. The video links the two. Until Phase 4, point times are estimated
from average point durations and that is stated in every report.

## Build phases

### Phase 0: Setup and data audit
Scaffold the repo, environment, CMake skeleton and CI. Parsers for Betfair BASIC
files and Sackmann point-by-point CSVs. `data/catalogue.csv` linking each match
across sources. **Done when:** the catalogue links ≥3 matches and one plot shows
Betfair implied probability over time with set scores marked.

### Phase 1: Markov win-probability engine
Given each player's serve-point win probability and the current score, return
P(win game), P(win set), P(win match). Handles tiebreaks, best-of-3/5 and
final-set rules. Game closed form:

    P_game(p) = p^4 (15 - 4p - 10p^2 / (1 - 2p(1-p)))      P_game(0.6) ≈ 0.736

**Done when:** the engine matches a Monte Carlo simulator within 0.005 across a
grid of p values and score states, and a plot shows P(match win) after every
point of a real match.

### Phase 2: Odds-only backtester
- Pre-match serve-point probabilities from tennis_atp stats (rolling 52 weeks,
  surface-adjusted with shrinkage; only matches before the target match).
- De-vig: implied = 1/odds, normalised so both runners sum to 1.
- Event-driven backtester: at decision time t the model uses only information up
  to t − delay. Bet when model − market exceeds a threshold. Flat or capped
  fractional Kelly. Commission is a parameter.
- Report Brier and log loss for model and market, calibration plot, P&L curve
  with bootstrap CIs and bet count.

**Done when:** a test proves there is no look-ahead (changing future data never
changes past decisions) and `reports/phase2.md` compares the model with the
market's own Brier score.

### Phase 3: Vision pipeline in Python
Court homography, player detection/tracking (YOLO + ByteTrack), pose, ball
tracking, point segmentation. Per-point features per player:

| Feature | Definition |
|---|---|
| distance_m | Total distance moved during the point |
| speed_mean, speed_peak | From smoothed court positions |
| speed_vs_baseline | Rolling peak speed over last 10 points ÷ first-set average |
| recovery_s | Time between point end and next serve |
| baseline_depth_m | Average distance behind the baseline |
| rally_shots | Shot count from ball or pose events |
| serve_pose_var | Variance in toss-arm and knee-bend keypoints across serves |

**Done when:** segmentation finds ≥90% of points on a hand-checked match,
homography reprojection error < 5 px, features saved per match as parquet.

### Phase 4: Aligning video, points and odds
Sequence-align detected video points to point-by-point rows; optional scoreboard
OCR; anchor video time to UTC via the in-play start. **Done when:** ≥95% of points
aligned and a merged parquet holds score state, features, clock time and market
probability.

### Phase 5: Point-win model
Logistic regression (score only), logistic regression (score + vision), gradient
boosting, small PyTorch GRU. Split by match, ordered by date. Log loss, Brier,
reliability diagram, ECE, calibration. **Done when:** `reports/phase5.md` has an
ablation table with CIs and an honest conclusion.

### Phase 6: Real-time C++ pipeline
Export models to ONNX; staged multithreaded pipeline with bounded queues
(decode → preprocess → inference → postprocess → tracking/homography → features).
p50/p95 latency per stage, fps, memory. Parity tests against Python. **Done
when:** ≥25 fps (or documented limits), parity tests pass, `reports/phase6.md`.

### Phase 7: Edge vs delay experiment
Grid: delay {60 s, 2 min, 5 min} × horizon {next game, next set, match} × model
{market, Markov score-only, Markov score+vision}. Brier improvement vs market,
simulated ROI with 95% CI, bet count. Main figure:
`reports/figures/edge_vs_delay.png`.

### Phase 8: Write-up and presentation
README in order: question, main figure, answer, architecture, results, how to
run, limitations, data credits. Mini-map demo GIF. justfile with one command per
phase.

## Evaluation protocol and pitfalls

The market's own Brier score is the baseline to beat.

| Pitfall | Guard |
|---|---|
| Look-ahead bias | Automated perturbation test; rolling features use only earlier points |
| Leaky splits | Split by match, ordered by date |
| Overfitting the strategy | Fix a final test set, touch it once, report every configuration tried |
| Small samples | Bootstrap CIs everywhere; state match and bet counts |
| Selection bias | Compare odds-only results on all matches vs the video subset |
| Unrealistic fills | Slippage penalty; Basic data has no order book |
| Overround | De-vig before every comparison |
| Broadcast delay | The delay variable in Phase 7; never assume zero |
