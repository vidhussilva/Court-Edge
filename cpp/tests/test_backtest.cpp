#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "courtedge/backtest.hpp"

using namespace courtedge;

namespace {
constexpr std::int64_t kMin = 60'000'000'000LL;

// A synthetic match: points every 40 s, a market tick every minute whose price drifts
// with the score (plus noise), so the strategy has something to trade on.
MatchData make_match(unsigned seed) {
  MatchData m;
  m.pa = 0.68;
  m.pb = 0.62;
  m.fmt = {3, FinalSet::Tiebreak};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0, 1);
  ScoreState s;
  std::int64_t t = 0;
  while (!s.is_over()) {
    const double q = s.server == 0 ? 0.62 : 1 - 0.66;  // "true" rates differ from the model's
    s = advance(s, u(rng) < q ? 0 : 1, m.fmt);
    t += 40'000'000'000LL;
    m.timeline.push_back({t, s});
  }
  m.winner = s.winner;
  const MarkovEngine truth(0.62, 0.66, m.fmt);
  size_t k = 0;
  ScoreState known;
  for (std::int64_t tm = -10 * kMin; tm <= t; tm += kMin) {
    while (k < m.timeline.size() && m.timeline[k].time_ns <= tm) known = m.timeline[k++].after;
    const double p = std::clamp(truth.win_probabilities(known).match + (u(rng) - 0.5) * 0.04,
                                0.02, 0.98);
    m.ticks.push_back({tm, to_tick(1 / p), to_tick(1 / (1 - p)), tm >= 0});
  }
  return m;
}

BacktestConfig active() {
  BacktestConfig c;
  c.threshold = 0.02;
  c.min_gap_s = 0;
  c.max_bets = 1000;
  c.delay_s = 120;
  return c;
}
}  // namespace

TEST(Market, DevigAndTicks) {
  EXPECT_NEAR(devig_multiplicative(1.8, 2.1), (1 / 1.8) / (1 / 1.8 + 1 / 2.1), 1e-15);
  EXPECT_DOUBLE_EQ(tick_down(2.0), 1.99);
  EXPECT_DOUBLE_EQ(tick_down(2.02), 2.0);
  EXPECT_DOUBLE_EQ(tick_down(4.1), 4.0);
  EXPECT_DOUBLE_EQ(tick_down(10.0), 9.8);
  EXPECT_DOUBLE_EQ(tick_down(1000), 990);
  EXPECT_DOUBLE_EQ(tick_down(1.01), 1.01);
  EXPECT_DOUBLE_EQ(to_tick(4.07), 4.1);
}

TEST(Market, KellyHandValues) {
  EXPECT_NEAR(kelly_fraction(0.6, 2.0, 0.0), 0.2, 1e-15);
  StakingConfig k;
  k.rule = StakingConfig::Rule::Kelly;
  k.kelly_cap = 0.5;
  EXPECT_NEAR(stake_for(0.6, 2.0, k, 0.0), 5.0, 1e-12);
}

TEST(Settlement, CommissionOnNetMarketWinnings) {
  const std::vector<BetRecord> bets{{0, 0, 2.0, 10, .5, .5}, {0, 1, 3.0, 5, .5, .5}};
  const auto a = settle_market(bets, 0, 0.05);
  EXPECT_NEAR(a.gross, 5.0, 1e-12);
  EXPECT_NEAR(a.net, 4.75, 1e-12);
  const auto b = settle_market(bets, 1, 0.05);
  EXPECT_NEAR(b.net, 0.0, 1e-12);
  EXPECT_EQ(b.commission, 0.0);
}

TEST(Backtest, NoLookAhead) {
  const MatchData m = make_match(3);
  const auto base = run_match(m, active());
  ASSERT_GT(base.bets.size(), 5u);
  const std::int64_t cut = m.ticks[m.ticks.size() / 2].time_ns;

  MatchData p = m;
  std::mt19937 rng(9);
  for (Tick& t : p.ticks)
    if (t.time_ns > cut) {
      t.odds_a = to_tick(1.1 + 10 * std::uniform_real_distribution<double>(0, 1)(rng));
      t.odds_b = to_tick(1.1 + 10 * std::uniform_real_distribution<double>(0, 1)(rng));
    }
  for (PointEvent& e : p.timeline)
    if (e.time_ns > cut) e.after = ScoreState{};  // wipe the future score
  p.winner = 1 - m.winner;
  const auto pert = run_match(p, active());

  for (size_t i = 0; i < base.predictions.size() && base.predictions[i].time_ns <= cut; ++i) {
    EXPECT_EQ(base.predictions[i].model.match, pert.predictions[i].model.match);
    EXPECT_EQ(base.predictions[i].market_p, pert.predictions[i].market_p);
  }
  for (size_t i = 0; i < base.bets.size() && base.bets[i].time_ns <= cut; ++i) {
    ASSERT_LT(i, pert.bets.size());
    EXPECT_EQ(base.bets[i].time_ns, pert.bets[i].time_ns);
    EXPECT_EQ(base.bets[i].side, pert.bets[i].side);
    EXPECT_EQ(base.bets[i].odds, pert.bets[i].odds);
  }
}

TEST(Backtest, DelayHidesRecentPoints) {
  const MatchData m = make_match(5);
  BacktestConfig c0 = active(), c5 = active();
  c0.delay_s = 0;
  c5.delay_s = 300;
  const MarkovEngine e(m.pa, m.pb, m.fmt);
  const auto p0 = predict(m, e, c0), p5 = predict(m, e, c5);
  for (size_t i = 0; i < p0.size(); ++i) {
    int n0 = 0, n5 = 0;
    for (const auto& ev : m.timeline) {
      n0 += ev.time_ns <= p0[i].time_ns;
      n5 += ev.time_ns <= p0[i].time_ns - 300 * 1'000'000'000LL;
    }
    EXPECT_EQ(p0[i].known_points, n0);
    EXPECT_EQ(p5[i].known_points, n5);
  }
}

TEST(Backtest, RunManyMatchesSequential) {
  std::vector<MatchData> ms;
  for (unsigned s = 0; s < 24; ++s) ms.push_back(make_match(s));
  const auto par = run_many(ms, active(), 4);
  for (size_t i = 0; i < ms.size(); ++i) {
    const auto seq = run_match(ms[i], active());
    EXPECT_EQ(par[i].bets.size(), seq.bets.size());
    EXPECT_DOUBLE_EQ(par[i].settlement.net, seq.settlement.net);
  }
}

TEST(Markov, RejectsInvalidStates) {
  const MarkovEngine e(0.65, 0.62, {3, FinalSet::Tiebreak});
  ScoreState s;
  s.games = {7, 7};
  EXPECT_THROW((void)e.win_probabilities(s), std::invalid_argument);
  s = {};
  s.points = {4, 4};
  EXPECT_THROW((void)e.win_probabilities(s), std::invalid_argument);
}
