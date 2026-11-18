#include "courtedge/market.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace courtedge {

namespace {
// (upper edge of band, increment); a band is (previous edge, upper edge].
constexpr std::array<std::pair<double, double>, 10> kLadder{{{2, 0.01},
                                                            {3, 0.02},
                                                            {4, 0.05},
                                                            {6, 0.1},
                                                            {10, 0.2},
                                                            {20, 0.5},
                                                            {30, 1},
                                                            {50, 2},
                                                            {100, 5},
                                                            {1000, 10}}};

double round2(double x) { return std::round(x * 100.0) / 100.0; }
}  // namespace

double devig_multiplicative(double odds_a, double odds_b) {
  if (odds_a <= 1.0 || odds_b <= 1.0) throw std::invalid_argument("decimal odds must be > 1");
  const double ia = 1.0 / odds_a, ib = 1.0 / odds_b;
  return ia / (ia + ib);
}

double to_tick(double price) {
  price = std::clamp(price, 1.01, 1000.0);
  double lo = 1.0;
  for (const auto& [hi, step] : kLadder) {
    if (price <= hi) return round2(lo + std::round((price - lo) / step) * step);
    lo = hi;
  }
  return 1000.0;
}

double tick_down(double price, int ticks) {
  double p = to_tick(price);
  for (int k = 0; k < ticks; ++k) {
    if (p <= 1.01) return 1.01;
    for (const auto& [hi, step] : kLadder)
      if (p <= hi + 1e-9) {
        p = round2(p - step);
        break;
      }
  }
  return p;
}

double kelly_fraction(double p, double odds, double commission) {
  const double b = (odds - 1.0) * (1.0 - commission);
  if (b <= 0) return 0.0;
  return std::max((p * b - (1.0 - p)) / b, 0.0);
}

double stake_for(double p, double odds, const StakingConfig& cfg, double commission) {
  if (cfg.rule == StakingConfig::Rule::Flat) return cfg.stake;
  const double f = std::min(cfg.kelly_fraction * kelly_fraction(p, odds, commission), cfg.kelly_cap);
  return cfg.bankroll * f;
}

}  // namespace courtedge
