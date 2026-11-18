#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "courtedge/score.hpp"

using namespace courtedge;

namespace {
ScoreState play(ScoreState s, const std::vector<int>& winners, const MatchFormat& f = {}) {
  for (int w : winners) s = advance(s, w, f);
  return s;
}
}  // namespace

TEST(Score, LoveGameSwitchesServer) {
  const auto s = play({}, {0, 0, 0, 0});
  EXPECT_EQ(s.games, (std::array<int, 2>{1, 0}));
  EXPECT_EQ(s.server, 1);
}

TEST(Score, DeuceIsNormalised) {
  auto s = play({}, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
  EXPECT_EQ(s.points, (std::array<int, 2>{3, 3}));
  EXPECT_EQ(play(s, {0}).points, (std::array<int, 2>{4, 3}));
  EXPECT_EQ(play(s, {1, 1}).games, (std::array<int, 2>{0, 1}));
}

TEST(Score, TiebreakServeOrderAndNextSetServer) {
  ScoreState s;
  s.games = {6, 5};
  s.server = 1;
  s = play(s, {1, 1, 1, 1});
  ASSERT_TRUE(s.tiebreak);
  EXPECT_EQ(s.server, 0);
  std::vector<int> servers;
  for (int w : {0, 1, 0, 1, 0, 1}) {
    servers.push_back(s.server);
    s = advance(s, w, {});
  }
  EXPECT_EQ(servers, (std::vector<int>{0, 1, 1, 0, 0, 1}));
  s = play(s, {0, 0, 0, 0, 0});
  EXPECT_EQ(s.sets, (std::array<int, 2>{1, 0}));
  EXPECT_EQ(s.server, 1);
}

TEST(Score, SuperTiebreakToTen) {
  const MatchFormat f{5, FinalSet::SuperTiebreak};
  ScoreState s;
  s.sets = {2, 2};
  s.games = {6, 6};
  s.tiebreak = true;
  s = play(s, std::vector<int>(9, 0), f);
  EXPECT_FALSE(s.is_over());
  s = advance(s, 0, f);
  EXPECT_EQ(s.winner, 0);
}

TEST(Score, AdvantageFinalSetHasNoTiebreak) {
  const MatchFormat f{3, FinalSet::Advantage};
  ScoreState s;
  s.sets = {1, 1};
  s.games = {6, 6};
  s = play(s, {0, 0, 0, 0, 1, 1, 1, 1}, f);
  EXPECT_FALSE(s.tiebreak);
  EXPECT_EQ(s.games, (std::array<int, 2>{7, 7}));
}

TEST(Score, ThrowsWhenOver) {
  ScoreState s;
  s.winner = 0;
  EXPECT_THROW(advance(s, 0, {}), std::invalid_argument);
  EXPECT_THROW(advance(ScoreState{}, 2, {}), std::invalid_argument);
}

TEST(Score, ParseFinalSetRoundTrip) {
  for (auto r : {FinalSet::Tiebreak, FinalSet::Advantage, FinalSet::Tiebreak12,
                 FinalSet::SuperTiebreak})
    EXPECT_EQ(parse_final_set(to_string(r)), r);
  EXPECT_THROW(parse_final_set("nope"), std::invalid_argument);
}
