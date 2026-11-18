#include "courtedge/simulate.hpp"

#include <algorithm>
#include <random>
#include <thread>
#include <vector>

namespace courtedge {

namespace {

struct Counts {
  std::int64_t game = 0, set = 0, match = 0;
};

Counts run_chunk(const ScoreState& start, double pa, double pb, const MatchFormat& fmt,
                 std::int64_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Counts c;
  for (std::int64_t i = 0; i < n; ++i) {
    ScoreState s = start;
    int game_winner = -1, set_winner = -1;
    while (!s.is_over()) {
      const double q = s.server == 0 ? pa : 1.0 - pb;
      const int w = u(rng) < q ? 0 : 1;
      const ScoreState next = advance(s, w, fmt);
      const bool game_ended = next.games != s.games || next.sets != s.sets || next.is_over();
      if (game_winner < 0 && game_ended) game_winner = w;
      if (set_winner < 0 && (next.sets != s.sets)) set_winner = w;
      s = next;
    }
    c.game += game_winner == 0;
    c.set += set_winner == 0;
    c.match += s.winner == 0;
  }
  return c;
}

}  // namespace

WinProbabilities simulate(const ScoreState& state, double pa, double pb, const MatchFormat& fmt,
                          const SimulationConfig& cfg) {
  if (state.is_over()) {
    const double v = state.winner == 0 ? 1.0 : 0.0;
    return {v, v, v};
  }
  unsigned threads = cfg.threads ? cfg.threads : std::max(1u, std::thread::hardware_concurrency());
  threads = static_cast<unsigned>(std::min<std::int64_t>(threads, std::max<std::int64_t>(cfg.n, 1)));

  std::vector<Counts> results(threads);
  {
    std::vector<std::jthread> pool;
    pool.reserve(threads);
    for (unsigned k = 0; k < threads; ++k) {
      const std::int64_t lo = cfg.n * k / threads, hi = cfg.n * (k + 1) / threads;
      // splitmix-style seed derivation keeps thread streams independent.
      const std::uint64_t seed = cfg.seed * 0x9E3779B97F4A7C15ULL + k + 1;
      pool.emplace_back([&, k, lo, hi, seed] {
        results[k] = run_chunk(state, pa, pb, fmt, hi - lo, seed);
      });
    }
  }  // jthreads join here

  Counts total;
  for (const auto& r : results) {
    total.game += r.game;
    total.set += r.set;
    total.match += r.match;
  }
  const auto n = static_cast<double>(cfg.n);
  return {static_cast<double>(total.game) / n, static_cast<double>(total.set) / n,
          static_cast<double>(total.match) / n};
}

}  // namespace courtedge
