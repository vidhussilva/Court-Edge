#pragma once

// Multithreaded Monte Carlo check of the Markov engine.

#include <cstdint>

#include "courtedge/markov.hpp"
#include "courtedge/score.hpp"

namespace courtedge {

struct SimulationConfig {
  std::int64_t n = 100'000;
  std::uint64_t seed = 0;
  unsigned threads = 0;  // 0 = std::thread::hardware_concurrency()
};

// Estimate P(A wins) the current game, set and match by playing the match out
// point by point with `advance`. Results are deterministic for a fixed
// (seed, threads) pair: thread k uses its own generator seeded from (seed, k).
WinProbabilities simulate(const ScoreState& state, double pa, double pb, const MatchFormat& fmt,
                          const SimulationConfig& cfg = {});

}  // namespace courtedge
