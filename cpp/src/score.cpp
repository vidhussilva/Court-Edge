#include "courtedge/score.hpp"

#include <algorithm>
#include <stdexcept>

namespace courtedge {

FinalSet parse_final_set(std::string_view name) {
  if (name == "tiebreak") return FinalSet::Tiebreak;
  if (name == "advantage") return FinalSet::Advantage;
  if (name == "tiebreak_12") return FinalSet::Tiebreak12;
  if (name == "super_tiebreak") return FinalSet::SuperTiebreak;
  throw std::invalid_argument("unknown final-set rule: " + std::string(name));
}

std::string_view to_string(FinalSet rule) {
  switch (rule) {
    case FinalSet::Tiebreak: return "tiebreak";
    case FinalSet::Advantage: return "advantage";
    case FinalSet::Tiebreak12: return "tiebreak_12";
    case FinalSet::SuperTiebreak: return "super_tiebreak";
  }
  return "?";
}

SetRule MatchFormat::set_rule(int sets_a, int sets_b) const noexcept {
  const bool is_final = sets_a == sets_b && sets_a == sets_to_win() - 1;
  if (!is_final) return {6, 7};
  switch (final_set) {
    case FinalSet::Tiebreak: return {6, 7};
    case FinalSet::Advantage: return {-1, 7};
    case FinalSet::Tiebreak12: return {12, 7};
    case FinalSet::SuperTiebreak: return {6, 10};
  }
  return {6, 7};
}

namespace {

// Once both players reach target-1 only the difference matters.
std::array<int, 2> normalise(std::array<int, 2> p, int target) {
  const int floor = target - 1;
  if (p[0] >= floor && p[1] >= floor) {
    const int diff = p[0] - p[1];
    return {floor + std::max(diff, 0), floor + std::max(-diff, 0)};
  }
  return p;
}

ScoreState set_won(ScoreState s, const MatchFormat& fmt, int w, int next_server) {
  ++s.sets[static_cast<size_t>(w)];
  s.points = {0, 0};
  s.tiebreak = false;
  if (s.sets[static_cast<size_t>(w)] == fmt.sets_to_win()) {
    s.winner = w;
    return s;
  }
  s.games = {0, 0};
  s.server = next_server;
  return s;
}

}  // namespace

ScoreState advance(const ScoreState& state, int point_winner, const MatchFormat& fmt) {
  if (state.is_over()) throw std::invalid_argument("match is already over");
  if (point_winner != 0 && point_winner != 1)
    throw std::invalid_argument("point_winner must be 0 or 1");

  const auto w = static_cast<size_t>(point_winner);
  const auto l = 1 - w;
  ScoreState s = state;
  std::array<int, 2> raw = s.points;
  ++raw[w];

  if (s.tiebreak) {
    const int target = fmt.set_rule(s.sets[0], s.sets[1]).tiebreak_points;
    if (raw[w] >= target && raw[w] - raw[l] >= 2) {
      ++s.games[w];
      return set_won(s, fmt, point_winner, 1 - s.tiebreak_first_server);
    }
    // Serve changes after the first point and then every two points.
    if ((raw[0] + raw[1]) % 2 == 1) s.server = 1 - s.server;
    s.points = normalise(raw, target);
    return s;
  }

  if (raw[w] >= 4 && raw[w] - raw[l] >= 2) {
    ++s.games[w];
    const int next_server = 1 - s.server;
    const int gw = s.games[w], gl = s.games[l];
    if (gw >= 6 && gw - gl >= 2) return set_won(s, fmt, point_winner, next_server);
    s.points = {0, 0};
    s.server = next_server;
    const SetRule rule = fmt.set_rule(s.sets[0], s.sets[1]);
    if (rule.tiebreak_at >= 0 && s.games[0] == s.games[1] && s.games[0] == rule.tiebreak_at) {
      s.tiebreak = true;
      s.tiebreak_first_server = next_server;
    }
    return s;
  }
  s.points = normalise(raw, 4);
  return s;
}

std::string describe(const ScoreState& s) {
  static constexpr const char* kNames[] = {"0", "15", "30", "40", "AD"};
  std::string pts = s.tiebreak ? "TB " + std::to_string(s.points[0]) + "-" +
                                     std::to_string(s.points[1])
                               : std::string(kNames[std::min(s.points[0], 4)]) + "-" +
                                     kNames[std::min(s.points[1], 4)];
  return "sets " + std::to_string(s.sets[0]) + "-" + std::to_string(s.sets[1]) + " games " +
         std::to_string(s.games[0]) + "-" + std::to_string(s.games[1]) + " " + pts +
         " (srv " + (s.server == 0 ? "A" : "B") + ")";
}

}  // namespace courtedge
