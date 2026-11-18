#pragma once

// Event-driven backtest loop (C++ port of courtedge.market.backtest).
//
// Same rules as the Python version: at each in-play tick t, the model sees the last
// score state with time <= t - delay, the market is the de-vigged last traded price
// at t, and a back bet is placed when |model - market| > threshold. The parity test
// (tests/test_core_parity.py) checks the two produce identical bets.

#include <cstdint>
#include <vector>

#include "courtedge/market.hpp"
#include "courtedge/markov.hpp"
#include "courtedge/score.hpp"

namespace courtedge {

struct Tick {
  std::int64_t time_ns;
  double odds_a, odds_b;
  bool in_play;
};

struct PointEvent {
  std::int64_t time_ns;  // when the point ended (state becomes known)
  ScoreState after;
};

struct BacktestConfig {
  double delay_s = 60.0;
  double threshold = 0.05;
  int slippage_ticks = 1;
  double min_gap_s = 300.0;
  int max_bets = 10;
  double min_odds = 1.05;
  double max_odds = 20.0;
  bool trade_preplay = false;
  StakingConfig staking{};
};

struct MatchData {
  std::vector<Tick> ticks;          // sorted by time
  std::vector<PointEvent> timeline;  // sorted by time
  double pa = 0.63, pb = 0.63;
  MatchFormat fmt{};
  int first_server = 0;
  int winner = 0;
  double commission = 0.05;
};

struct Prediction {
  std::int64_t time_ns;
  double market_p;
  WinProbabilities model;
  int known_points;
};

struct BetRecord {
  std::int64_t time_ns;
  int side;
  double odds, stake, model_p, market_p;
};

struct Settlement {
  int n_bets = 0;
  double staked = 0, gross = 0, commission = 0, net = 0;
};

struct MatchResult {
  std::vector<Prediction> predictions;
  std::vector<BetRecord> bets;
  Settlement settlement;
};

std::vector<Prediction> predict(const MatchData& m, const MarkovEngine& engine,
                                const BacktestConfig& cfg);
std::vector<BetRecord> decide(const std::vector<Prediction>& preds, const MatchData& m,
                              const BacktestConfig& cfg);
Settlement settle_market(const std::vector<BetRecord>& bets, int winner, double commission);
MatchResult run_match(const MatchData& m, const BacktestConfig& cfg);

// Run many matches on a pool of threads. Matches are handed out through an atomic
// counter, so a long five-setter doesn't hold up a whole chunk of short matches.
// Results come back in input order.
std::vector<MatchResult> run_many(const std::vector<MatchData>& matches, const BacktestConfig& cfg,
                                  unsigned threads = 0);

}  // namespace courtedge
