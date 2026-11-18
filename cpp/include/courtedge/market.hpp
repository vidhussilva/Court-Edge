#pragma once

// Market helpers shared by the backtester: de-vigging, the Betfair price ladder, staking.

namespace courtedge {

// P(A wins) from both runners' decimal odds, normalised so the pair sums to 1.
double devig_multiplicative(double odds_a, double odds_b);

// Nearest valid Betfair price, clamped to [1.01, 1000].
double to_tick(double price);
// Move a price `ticks` steps down the ladder (a worse price for a backer).
double tick_down(double price, int ticks = 1);

struct StakingConfig {
  enum class Rule { Flat, Kelly } rule = Rule::Flat;
  double stake = 1.0;
  double bankroll = 100.0;
  double kelly_fraction = 0.25;
  double kelly_cap = 0.02;
};

double kelly_fraction(double p, double odds, double commission);
double stake_for(double p, double odds, const StakingConfig& cfg, double commission);

}  // namespace courtedge
