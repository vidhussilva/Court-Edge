#include "courtedge/backtest.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>

namespace courtedge {

std::vector<Prediction> predict(const MatchData& m, const MarkovEngine& engine,
                                const BacktestConfig& cfg) {
  std::vector<Prediction> out;
  out.reserve(m.ticks.size());
  ScoreState initial;
  initial.server = m.first_server;
  const auto delay_ns = static_cast<std::int64_t>(std::llround(cfg.delay_s * 1e9));
  // Ticks and points are both sorted, so the "last known point" pointer only ever
  // moves forward: one O(ticks + points) sweep instead of a binary search per tick.
  size_t next = 0;
  for (const Tick& t : m.ticks) {
    if (!cfg.trade_preplay && !t.in_play) continue;
    const std::int64_t visible = t.time_ns - delay_ns;
    while (next < m.timeline.size() && m.timeline[next].time_ns <= visible) ++next;
    const ScoreState& state = next == 0 ? initial : m.timeline[next - 1].after;
    out.push_back({t.time_ns, devig_multiplicative(t.odds_a, t.odds_b),
                   engine.win_probabilities(state), static_cast<int>(next)});
  }
  return out;
}

std::vector<BetRecord> decide(const std::vector<Prediction>& preds, const MatchData& m,
                              const BacktestConfig& cfg) {
  std::vector<BetRecord> bets;
  const auto gap_ns = static_cast<std::int64_t>(std::llround(cfg.min_gap_s * 1e9));
  bool have_last = false;
  std::int64_t last = 0;
  size_t tick = 0;
  for (const Prediction& p : preds) {
    // Recover the raw odds for this prediction's tick (same time stamp).
    while (tick < m.ticks.size() && m.ticks[tick].time_ns < p.time_ns) ++tick;
    const Tick& t = m.ticks[tick];
    if (static_cast<int>(bets.size()) >= cfg.max_bets) break;
    if (have_last && p.time_ns - last < gap_ns) continue;
    const double edge_a = p.model.match - p.market_p;
    if (std::abs(edge_a) <= cfg.threshold) continue;
    const int side = edge_a > 0 ? 0 : 1;
    const double raw = side == 0 ? t.odds_a : t.odds_b;
    if (raw < cfg.min_odds || raw > cfg.max_odds) continue;
    const double odds = tick_down(raw, cfg.slippage_ticks);
    const double model_p = side == 0 ? p.model.match : 1 - p.model.match;
    const double market_p = side == 0 ? p.market_p : 1 - p.market_p;
    const double stake = stake_for(model_p, odds, cfg.staking, m.commission);
    if (stake <= 0) continue;
    bets.push_back({p.time_ns, side, odds, stake, model_p, market_p});
    have_last = true;
    last = p.time_ns;
  }
  return bets;
}

Settlement settle_market(const std::vector<BetRecord>& bets, int winner, double commission) {
  Settlement s;
  for (const BetRecord& b : bets) {
    s.gross += b.side == winner ? b.stake * (b.odds - 1.0) : -b.stake;
    s.staked += b.stake;
  }
  s.n_bets = static_cast<int>(bets.size());
  // Betfair charges commission on net market winnings only.
  s.commission = s.gross > 0 ? commission * s.gross : 0.0;
  s.net = s.gross - s.commission;
  return s;
}

MatchResult run_match(const MatchData& m, const BacktestConfig& cfg) {
  const MarkovEngine engine(m.pa, m.pb, m.fmt);
  MatchResult r;
  r.predictions = predict(m, engine, cfg);
  r.bets = decide(r.predictions, m, cfg);
  r.settlement = settle_market(r.bets, m.winner, m.commission);
  return r;
}

std::vector<MatchResult> run_many(const std::vector<MatchData>& matches, const BacktestConfig& cfg,
                                  unsigned threads) {
  std::vector<MatchResult> out(matches.size());
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  threads = static_cast<unsigned>(std::min<size_t>(threads, std::max<size_t>(matches.size(), 1)));
  std::atomic<size_t> next{0};
  {
    std::vector<std::jthread> pool;
    for (unsigned k = 0; k < threads; ++k)
      pool.emplace_back([&] {
        for (size_t i = next.fetch_add(1); i < matches.size(); i = next.fetch_add(1))
          out[i] = run_match(matches[i], cfg);  // each slot written by exactly one thread
      });
  }
  return out;
}

}  // namespace courtedge
