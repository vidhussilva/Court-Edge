#include <benchmark/benchmark.h>

#include <random>

#include "courtedge/backtest.hpp"

using namespace courtedge;

namespace {
std::vector<MatchData> make_matches(int n) {
  std::vector<MatchData> out;
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(0, 1);
  for (int i = 0; i < n; ++i) {
    MatchData m;
    m.fmt = {5, FinalSet::SuperTiebreak};
    m.pa = 0.6 + 0.1 * u(rng);
    m.pb = 0.6 + 0.1 * u(rng);
    ScoreState s;
    std::int64_t t = 0;
    while (!s.is_over()) {
      s = advance(s, u(rng) < (s.server == 0 ? m.pa : 1 - m.pb) ? 0 : 1, m.fmt);
      t += 40'000'000'000LL;
      m.timeline.push_back({t, s});
    }
    m.winner = s.winner;
    for (std::int64_t tm = 0; tm <= t; tm += 60'000'000'000LL) {
      const double p = 0.2 + 0.6 * u(rng);
      m.ticks.push_back({tm, to_tick(1 / p), to_tick(1 / (1 - p)), true});
    }
    out.push_back(std::move(m));
  }
  return out;
}
}  // namespace

// 200 best-of-5 matches (about 40k in-play ticks), the size of one Phase 7 grid cell.
static void BM_RunMany(benchmark::State& st) {
  static const auto matches = make_matches(200);
  BacktestConfig cfg;
  for (auto _ : st)
    benchmark::DoNotOptimize(run_many(matches, cfg, static_cast<unsigned>(st.range(0))));
}
BENCHMARK(BM_RunMany)->Arg(1)->Arg(2)->Arg(4)->Unit(benchmark::kMillisecond)->UseRealTime();
