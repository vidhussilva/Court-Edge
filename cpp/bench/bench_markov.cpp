#include <benchmark/benchmark.h>

#include "courtedge/markov.hpp"
#include "courtedge/simulate.hpp"

using namespace courtedge;

// Building the tables for one (pa, pb) pair: paid once per match in a backtest.
static void BM_EngineConstruct(benchmark::State& st) {
  const MatchFormat f{5, FinalSet::SuperTiebreak};
  double pa = 0.60;
  for (auto _ : st) {
    MarkovEngine e(pa, 0.62, f);
    benchmark::DoNotOptimize(e.p_match());
    pa = pa < 0.75 ? pa + 1e-6 : 0.60;
  }
}
BENCHMARK(BM_EngineConstruct);

// One query: paid at every decision time.
static void BM_WinProbabilitiesQuery(benchmark::State& st) {
  const MarkovEngine e(0.66, 0.62, {5, FinalSet::SuperTiebreak});
  ScoreState s;
  s.sets = {1, 2};
  s.games = {4, 5};
  s.points = {2, 3};
  for (auto _ : st) benchmark::DoNotOptimize(e.win_probabilities(s));
}
BENCHMARK(BM_WinProbabilitiesQuery);

static void BM_Simulate100k(benchmark::State& st) {
  const auto threads = static_cast<unsigned>(st.range(0));
  for (auto _ : st)
    benchmark::DoNotOptimize(simulate({}, 0.66, 0.62, {5, FinalSet::SuperTiebreak},
                                      {100'000, 1, threads}));
}
BENCHMARK(BM_Simulate100k)->Arg(1)->Arg(2)->Arg(4)->Unit(benchmark::kMillisecond)->UseRealTime();
