#pragma once

// Tennis score state machine. Mirrors src/courtedge/markov/score.py exactly
// (same normalisation of deuce and tiebreak scores) so states can be compared
// across languages in the parity tests.

#include <array>
#include <string>
#include <string_view>

namespace courtedge {

enum class FinalSet { Tiebreak, Advantage, Tiebreak12, SuperTiebreak };

FinalSet parse_final_set(std::string_view name);
std::string_view to_string(FinalSet rule);

struct SetRule {
  int tiebreak_at;      // games each at which a tiebreak starts; -1 = advantage set
  int tiebreak_points;  // points needed to win the tiebreak (win by two)
  bool operator==(const SetRule&) const = default;
};

struct MatchFormat {
  int best_of = 3;
  FinalSet final_set = FinalSet::Tiebreak;

  [[nodiscard]] int sets_to_win() const noexcept { return best_of / 2 + 1; }
  [[nodiscard]] SetRule set_rule(int sets_a, int sets_b) const noexcept;
};

struct ScoreState {
  std::array<int, 2> sets{0, 0};
  std::array<int, 2> games{0, 0};
  std::array<int, 2> points{0, 0};
  int server = 0;                 // player serving the next point
  bool tiebreak = false;
  int tiebreak_first_server = 0;  // only meaningful while tiebreak is true
  int winner = -1;                // -1 while the match is in progress

  [[nodiscard]] bool is_over() const noexcept { return winner >= 0; }
  bool operator==(const ScoreState&) const = default;
};

// State after `point_winner` (0 or 1) wins the next point.
// Throws std::invalid_argument if the match is already over.
ScoreState advance(const ScoreState& state, int point_winner, const MatchFormat& fmt);

std::string describe(const ScoreState& state);

}  // namespace courtedge
