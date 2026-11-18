#pragma once

// Exact Markov win-probability engine (C++ port of courtedge.markov.engine).
//
// Construction fills every memo table for one (pa, pb, format) triple; after that
// the object is immutable, so one engine can be shared across threads and every
// query is a handful of table lookups.

#include <array>
#include <vector>

#include "courtedge/score.hpp"

namespace courtedge {

struct WinProbabilities {
  double game;   // P(A wins the current game; a tiebreak counts as the game)
  double set;    // P(A wins the current set)
  double match;  // P(A wins the match)
};

// P(server holds) from 0-0, closed form.
double p_game(double p);

class MarkovEngine {
 public:
  // pa = P(A wins a point on A's serve), pb = P(B wins a point on B's serve).
  MarkovEngine(double pa, double pb, MatchFormat fmt);

  [[nodiscard]] WinProbabilities win_probabilities(const ScoreState& state) const;
  [[nodiscard]] double p_match(int first_server = 0) const;

  [[nodiscard]] double pa() const noexcept { return pa_; }
  [[nodiscard]] double pb() const noexcept { return pb_; }
  [[nodiscard]] const MatchFormat& format() const noexcept { return fmt_; }

  // (A wins, next server 0), (A wins, 1), (B wins, 0), (B wins, 1)
  using SetOutcome = std::array<double, 4>;

 private:
  static constexpr int kMaxGames = 14;   // covers tiebreak-at-12 sets
  static constexpr int kMaxTbPts = 11;   // covers 10-point tiebreaks
  static constexpr int kMaxSets = 3;

  double pa_, pb_;
  MatchFormat fmt_;

  // hold_[s][a][b]: P(server s wins the game) with server on a points, returner on b.
  std::array<std::array<std::array<double, 5>, 5>, 2> hold_{};
  // tb_[t][a][b][server]: P(A wins tiebreak), t=0 for 7-point, t=1 for 10-point.
  std::array<std::array<std::array<std::array<double, 2>, kMaxTbPts + 1>, kMaxTbPts + 1>, 2> tb_{};
  // set_[r][ga][gb][server], r=0 standard set, r=1 deciding set.
  std::vector<SetOutcome> set_;
  std::vector<char> set_done_;
  // match_[sa][sb][server]: P(A wins match) at the start of a set.
  std::array<std::array<std::array<double, 2>, kMaxSets + 1>, kMaxSets + 1> match_{};

  [[nodiscard]] double point_a(int server) const noexcept { return server == 0 ? pa_ : 1 - pb_; }
  [[nodiscard]] double game_a(const std::array<int, 2>& pts, int server) const noexcept;
  [[nodiscard]] double tiebreak_a(int a, int b, int server, int target) const noexcept;

  double fill_tiebreak(int t, int target, int a, int b, int server, std::vector<char>& done);
  const SetOutcome& fill_set(int r, const SetRule& rule, int ga, int gb, int server);
  [[nodiscard]] SetOutcome after_game(int r, const SetRule& rule, int ga, int gb,
                                      int server, double g);
  [[nodiscard]] SetOutcome after_game_const(int r, int ga, int gb, int server, double g) const;
  [[nodiscard]] const SetOutcome& set_at(int r, int ga, int gb, int server) const;
  [[nodiscard]] int rule_index(int sa, int sb) const noexcept;
  [[nodiscard]] static std::array<int, 2> normalise_games(int ga, int gb, const SetRule& rule);
};

}  // namespace courtedge
