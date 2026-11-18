#include "courtedge/markov.hpp"

#include <algorithm>
#include <cmath>

namespace courtedge {

namespace {

constexpr double kEps = 1e-12;

double clamp_p(double p) { return std::clamp(p, kEps, 1.0 - kEps); }

// P(server wins the game) from (a, b), normalised score (deuce = 3-3).
double game_from(double p, int a, int b) {
  if (a >= 4 && a - b >= 2) return 1.0;
  if (b >= 4 && b - a >= 2) return 0.0;
  if (a >= 3 && b >= 3) {
    const double deuce = p * p / (p * p + (1 - p) * (1 - p));
    if (a == b) return deuce;
    return a > b ? p + (1 - p) * deuce : p * deuce;
  }
  return p * game_from(p, a + 1, b) + (1 - p) * game_from(p, a, b + 1);
}

int next_tb_server(int a, int b, int server) {
  // After the point that brings the total to a+b+1: switch on odd totals.
  return (a + b + 1) % 2 == 1 ? 1 - server : server;
}

}  // namespace

double p_game(double p) {
  p = clamp_p(p);
  return std::pow(p, 4) * (15 - 4 * p - 10 * p * p / (1 - 2 * p * (1 - p)));
}

MarkovEngine::MarkovEngine(double pa, double pb, MatchFormat fmt)
    : pa_(clamp_p(pa)), pb_(clamp_p(pb)), fmt_(fmt) {
  for (int s = 0; s < 2; ++s) {
    const double p = s == 0 ? pa_ : pb_;
    for (int a = 0; a <= 4; ++a)
      for (int b = 0; b <= 4; ++b)
        hold_[static_cast<size_t>(s)][static_cast<size_t>(a)][static_cast<size_t>(b)] =
            game_from(p, a, b);
  }

  for (int t = 0; t < 2; ++t) {
    const int target = t == 0 ? 7 : 10;
    std::vector<char> done(static_cast<size_t>((kMaxTbPts + 1) * (kMaxTbPts + 1) * 2), 0);
    for (int a = 0; a <= target; ++a)
      for (int b = 0; b <= target; ++b)
        for (int s = 0; s < 2; ++s) fill_tiebreak(t, target, a, b, s, done);
  }

  set_.assign(static_cast<size_t>(2 * kMaxGames * kMaxGames * 2), SetOutcome{});
  set_done_.assign(set_.size(), 0);
  const SetRule standard{6, 7};
  const SetRule deciding = fmt_.set_rule(fmt_.sets_to_win() - 1, fmt_.sets_to_win() - 1);
  for (int r = 0; r < 2; ++r) {
    const SetRule& rule = r == 0 ? standard : deciding;
    for (int ga = kMaxGames - 1; ga >= 0; --ga)
      for (int gb = kMaxGames - 1; gb >= 0; --gb)
        for (int s = 0; s < 2; ++s) {
          const auto n = normalise_games(ga, gb, rule);
          if (n[0] == ga && n[1] == gb) fill_set(r, rule, ga, gb, s);
        }
  }

  const int need = fmt_.sets_to_win();
  for (int sa = need; sa >= 0; --sa)
    for (int sb = need; sb >= 0; --sb)
      for (int s = 0; s < 2; ++s) {
        double& m = match_[static_cast<size_t>(sa)][static_cast<size_t>(sb)][static_cast<size_t>(s)];
        if (sa == need && sb == need) continue;
        if (sa == need) { m = 1.0; continue; }
        if (sb == need) { m = 0.0; continue; }
        const SetOutcome& o = set_at(rule_index(sa, sb), 0, 0, s);
        auto M = [&](int x, int y, int srv) {
          return match_[static_cast<size_t>(x)][static_cast<size_t>(y)][static_cast<size_t>(srv)];
        };
        m = o[0] * M(sa + 1, sb, 0) + o[1] * M(sa + 1, sb, 1) + o[2] * M(sa, sb + 1, 0) +
            o[3] * M(sa, sb + 1, 1);
      }
}

double MarkovEngine::fill_tiebreak(int t, int target, int a, int b, int server,
                                   std::vector<char>& done) {
  const auto ua = static_cast<size_t>(a), ub = static_cast<size_t>(b),
             us = static_cast<size_t>(server), ut = static_cast<size_t>(t);
  const size_t key = (ua * (kMaxTbPts + 1) + ub) * 2 + us;
  double& out = tb_[ut][ua][ub][us];
  if (done[key]) return out;
  done[key] = 1;

  if (a >= target && a - b >= 2) return out = 1.0;
  if (b >= target && b - a >= 2) return out = 0.0;
  if (a >= target - 1 && b >= target - 1) {
    // Level at target-1: the pair is served by `server` then the other player and
    // the serve then stays with the other player (see the Python engine).
    auto pair = [&](int s) {
      const double q1 = point_a(s), q2 = point_a(1 - s);
      return std::array<double, 2>{q1 * q2, q1 * (1 - q2) + (1 - q1) * q2};
    };
    if (a == b) {
      const auto [ws, ts] = pair(server);
      const auto [wo, to] = pair(1 - server);
      return out = (ws + ts * wo) / (1 - ts * to);
    }
    const double q = point_a(server);
    const int nxt = next_tb_server(a, b, server);
    const double level = fill_tiebreak(t, target, target - 1, target - 1, nxt, done);
    return out = a > b ? q + (1 - q) * level : q * level;
  }
  const double q = point_a(server);
  const int nxt = next_tb_server(a, b, server);
  return out = q * fill_tiebreak(t, target, a + 1, b, nxt, done) +
               (1 - q) * fill_tiebreak(t, target, a, b + 1, nxt, done);
}

std::array<int, 2> MarkovEngine::normalise_games(int ga, int gb, const SetRule& rule) {
  // Advantage sets can run to 70-68; once both reach 6 only the difference matters.
  if (rule.tiebreak_at < 0 && ga >= 6 && gb >= 6) {
    const int d = ga - gb;
    return {6 + std::max(d, 0), 6 + std::max(-d, 0)};
  }
  return {ga, gb};
}

int MarkovEngine::rule_index(int sa, int sb) const noexcept {
  const int last = fmt_.sets_to_win() - 1;
  return (sa == last && sb == last) ? 1 : 0;
}

const MarkovEngine::SetOutcome& MarkovEngine::set_at(int r, int ga, int gb, int server) const {
  const size_t key =
      ((static_cast<size_t>(r) * kMaxGames + static_cast<size_t>(ga)) * kMaxGames +
       static_cast<size_t>(gb)) * 2 + static_cast<size_t>(server);
  return set_[key];
}

MarkovEngine::SetOutcome MarkovEngine::after_game(int r, const SetRule& rule, int ga, int gb,
                                                  int server, double g) {
  SetOutcome total{};
  const int nxt = 1 - server;
  for (int winner = 0; winner < 2; ++winner) {
    const double prob = winner == 0 ? g : 1 - g;
    const int na = ga + (winner == 0), nb = gb + (winner == 1);
    const int hi = std::max(na, nb), lo = std::min(na, nb);
    if (hi >= 6 && hi - lo >= 2) {
      total[static_cast<size_t>(winner * 2 + nxt)] += prob;
    } else {
      const auto n = normalise_games(na, nb, rule);
      const SetOutcome& o = fill_set(r, rule, n[0], n[1], nxt);
      for (size_t k = 0; k < 4; ++k) total[k] += prob * o[k];
    }
  }
  return total;
}

MarkovEngine::SetOutcome MarkovEngine::after_game_const(int r, int ga, int gb, int server,
                                                        double g) const {
  const SetRule rule = r == 0 ? SetRule{6, 7}
                              : fmt_.set_rule(fmt_.sets_to_win() - 1, fmt_.sets_to_win() - 1);
  SetOutcome total{};
  const int nxt = 1 - server;
  for (int winner = 0; winner < 2; ++winner) {
    const double prob = winner == 0 ? g : 1 - g;
    const int na = ga + (winner == 0), nb = gb + (winner == 1);
    const int hi = std::max(na, nb), lo = std::min(na, nb);
    if (hi >= 6 && hi - lo >= 2) {
      total[static_cast<size_t>(winner * 2 + nxt)] += prob;
    } else {
      const auto n = normalise_games(na, nb, rule);
      const SetOutcome& o = set_at(r, n[0], n[1], nxt);
      for (size_t k = 0; k < 4; ++k) total[k] += prob * o[k];
    }
  }
  return total;
}

const MarkovEngine::SetOutcome& MarkovEngine::fill_set(int r, const SetRule& rule, int ga,
                                                       int gb, int server) {
  const size_t key =
      ((static_cast<size_t>(r) * kMaxGames + static_cast<size_t>(ga)) * kMaxGames +
       static_cast<size_t>(gb)) * 2 + static_cast<size_t>(server);
  SetOutcome& out = set_[key];
  if (set_done_[key]) return out;
  set_done_[key] = 1;

  if (rule.tiebreak_at >= 0 && ga == rule.tiebreak_at && gb == rule.tiebreak_at) {
    const double p = tiebreak_a(0, 0, server, rule.tiebreak_points);
    out = {};
    out[static_cast<size_t>(0 * 2 + (1 - server))] = p;
    out[static_cast<size_t>(1 * 2 + (1 - server))] = 1 - p;
    return out;
  }
  // Past the tiebreak score the set is already over: unreachable, leave zeros.
  if (rule.tiebreak_at >= 0 && std::max(ga, gb) > rule.tiebreak_at) return out;
  const double g = game_a({0, 0}, server);
  if (rule.tiebreak_at < 0 && ga == gb && ga >= 6) {
    const double g2 = game_a({0, 0}, 1 - server);
    const double both_a = g * g2, both_b = (1 - g) * (1 - g2);
    const double p = both_a / (both_a + both_b);
    out = {};
    out[static_cast<size_t>(server)] = p;
    out[static_cast<size_t>(2 + server)] = 1 - p;
    return out;
  }
  out = after_game(r, rule, ga, gb, server, g);
  return out;
}

double MarkovEngine::game_a(const std::array<int, 2>& pts, int server) const noexcept {
  const auto a = static_cast<size_t>(pts[0]), b = static_cast<size_t>(pts[1]);
  return server == 0 ? hold_[0][a][b] : 1.0 - hold_[1][b][a];
}

double MarkovEngine::tiebreak_a(int a, int b, int server, int target) const noexcept {
  const size_t t = target == 7 ? 0 : 1;
  return tb_[t][static_cast<size_t>(a)][static_cast<size_t>(b)][static_cast<size_t>(server)];
}

WinProbabilities MarkovEngine::win_probabilities(const ScoreState& s) const {
  if (s.is_over()) {
    const double v = s.winner == 0 ? 1.0 : 0.0;
    return {v, v, v};
  }
  const int r = rule_index(s.sets[0], s.sets[1]);
  const SetRule rule = fmt_.set_rule(s.sets[0], s.sets[1]);
  double g;
  SetOutcome out{};
  if (s.tiebreak) {
    g = tiebreak_a(s.points[0], s.points[1], s.server, rule.tiebreak_points);
    const int nxt = 1 - s.tiebreak_first_server;
    out[static_cast<size_t>(nxt)] = g;
    out[static_cast<size_t>(2 + nxt)] = 1 - g;
  } else {
    g = game_a(s.points, s.server);
    const auto n = normalise_games(s.games[0], s.games[1], rule);
    out = after_game_const(r, n[0], n[1], s.server, g);
  }
  const auto sa = static_cast<size_t>(s.sets[0]), sb = static_cast<size_t>(s.sets[1]);
  const double match = out[0] * match_[sa + 1][sb][0] + out[1] * match_[sa + 1][sb][1] +
                       out[2] * match_[sa][sb + 1][0] + out[3] * match_[sa][sb + 1][1];
  return {g, out[0] + out[1], match};
}

double MarkovEngine::p_match(int first_server) const {
  return match_[0][0][static_cast<size_t>(first_server)];
}

}  // namespace courtedge
