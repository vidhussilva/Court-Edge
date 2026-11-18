#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "courtedge/markov.hpp"
#include "courtedge/simulate.hpp"

using namespace courtedge;

TEST(Markov, GameClosedForm) {
  EXPECT_NEAR(p_game(0.6), 0.7357, 1e-4);
  EXPECT_NEAR(p_game(0.5), 0.5, 1e-12);
  const MarkovEngine e(0.6, 0.6, {});
  EXPECT_NEAR(e.win_probabilities({}).game, p_game(0.6), 1e-12);
}

TEST(Markov, EqualPlayersAreCoinFlips) {
  for (auto r : {FinalSet::Tiebreak, FinalSet::Advantage, FinalSet::Tiebreak12,
                 FinalSet::SuperTiebreak})
    for (int bo : {3, 5}) EXPECT_NEAR(MarkovEngine(0.63, 0.63, {bo, r}).p_match(), 0.5, 1e-12);
}

TEST(Markov, Symmetry) {
  const MatchFormat f{5, FinalSet::Advantage};
  EXPECT_NEAR(MarkovEngine(0.67, 0.61, f).p_match(0),
              1.0 - MarkovEngine(0.61, 0.67, f).p_match(1), 1e-12);
}

TEST(Markov, MartingaleAlongRandomWalk) {
  const MatchFormat f{5, FinalSet::Tiebreak12};
  const MarkovEngine e(0.7, 0.68, f);
  ScoreState s;
  s.sets = {2, 2};
  unsigned x = 12345;
  for (int i = 0; i < 600 && !s.is_over(); ++i) {
    const double q = s.server == 0 ? e.pa() : 1 - e.pb();
    const auto a = advance(s, 0, f), b = advance(s, 1, f);
    EXPECT_NEAR(e.win_probabilities(s).match,
                q * e.win_probabilities(a).match + (1 - q) * e.win_probabilities(b).match,
                1e-12);
    x = x * 1103515245u + 12345u;
    s = (x >> 16) % 2 ? a : b;
  }
}

// Every row of tests/fixtures/markov_reference.csv was produced by the Python engine.
TEST(Markov, ParityWithPythonEngine) {
  std::ifstream in("tests/fixtures/markov_reference.csv");
  ASSERT_TRUE(in) << "run ctest from the repo root (WORKING_DIRECTORY is set in CMake)";
  std::string line;
  std::getline(in, line);
  int rows = 0;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> c;
    for (std::string cell; std::getline(ss, cell, ',');) c.push_back(cell);
    ASSERT_EQ(c.size(), 16u);
    const MatchFormat f{std::stoi(c[2]), parse_final_set(c[3])};
    const MarkovEngine e(std::stod(c[0]), std::stod(c[1]), f);
    ScoreState s;
    s.sets = {std::stoi(c[4]), std::stoi(c[5])};
    s.games = {std::stoi(c[6]), std::stoi(c[7])};
    s.points = {std::stoi(c[8]), std::stoi(c[9])};
    s.server = std::stoi(c[10]);
    s.tiebreak = c[11] == "1";
    s.tiebreak_first_server = std::stoi(c[12]);
    const auto w = e.win_probabilities(s);
    EXPECT_NEAR(w.game, std::stod(c[13]), 1e-10) << line;
    EXPECT_NEAR(w.set, std::stod(c[14]), 1e-10) << line;
    EXPECT_NEAR(w.match, std::stod(c[15]), 1e-10) << line;
    ++rows;
  }
  EXPECT_GT(rows, 1000);
}

class MonteCarlo : public ::testing::TestWithParam<double> {};

TEST_P(MonteCarlo, EngineMatchesSimulation) {
  const double p = GetParam();
  const MatchFormat f{5, FinalSet::SuperTiebreak};
  ScoreState deuce;
  deuce.games = {4, 5};
  deuce.points = {3, 3};
  deuce.server = 1;
  for (const ScoreState& s : {ScoreState{}, deuce}) {
    const auto exact = MarkovEngine(p, 0.62, f).win_probabilities(s);
    // 400k sims: SE <= 0.0008, so the spec tolerance of 0.005 sits at about 6 sigma
    // and 36 comparisons do not fail by chance.
    const auto mc = simulate(s, p, 0.62, f, {400'000, 7, 0});
    EXPECT_NEAR(exact.game, mc.game, 0.005);
    EXPECT_NEAR(exact.set, mc.set, 0.005);
    EXPECT_NEAR(exact.match, mc.match, 0.005);
  }
}

INSTANTIATE_TEST_SUITE_P(Grid, MonteCarlo, ::testing::Values(0.5, 0.55, 0.6, 0.65, 0.7, 0.75));

TEST(Simulate, DeterministicForFixedSeedAndThreads) {
  const auto a = simulate({}, 0.64, 0.6, {}, {20'000, 3, 4});
  const auto b = simulate({}, 0.64, 0.6, {}, {20'000, 3, 4});
  EXPECT_EQ(a.match, b.match);
}
