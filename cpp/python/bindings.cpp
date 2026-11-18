// Python bindings: `from courtedge import _core`.
// Optional. The pure-Python package works without them; tests check parity when built.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "courtedge/backtest.hpp"
#include "courtedge/markov.hpp"
#include "courtedge/score.hpp"
#include "courtedge/simulate.hpp"
#include "courtedge/version.hpp"

namespace py = pybind11;
using namespace courtedge;

namespace {
MatchFormat make_format(int best_of, const std::string& final_set) {
  return {best_of, parse_final_set(final_set)};
}

ScoreState make_state(std::array<int, 2> sets, std::array<int, 2> games,
                      std::array<int, 2> points, int server, bool tiebreak,
                      int tiebreak_first_server, int winner) {
  ScoreState s;
  s.sets = sets;
  s.games = games;
  s.points = points;
  s.server = server;
  s.tiebreak = tiebreak;
  s.tiebreak_first_server = tiebreak_first_server;
  s.winner = winner;
  return s;
}
// Build a MatchData from numpy arrays. `states` is an (n, 10) int array with columns
// sets_a, sets_b, games_a, games_b, points_a, points_b, server, tiebreak,
// tb_first_server, match_winner (-1 while in progress).
MatchData make_match(py::array_t<std::int64_t> tick_ns, py::array_t<double> odds_a,
                     py::array_t<double> odds_b, py::array_t<bool> in_play,
                     py::array_t<std::int64_t> point_ns, py::array_t<int> states, double pa,
                     double pb, const MatchFormat& fmt, int first_server, int winner,
                     double commission) {
  MatchData m;
  m.pa = pa;
  m.pb = pb;
  m.fmt = fmt;
  m.first_server = first_server;
  m.winner = winner;
  m.commission = commission;
  auto t = tick_ns.unchecked<1>();
  auto a = odds_a.unchecked<1>();
  auto b = odds_b.unchecked<1>();
  auto ip = in_play.unchecked<1>();
  for (py::ssize_t i = 0; i < t.shape(0); ++i) m.ticks.push_back({t(i), a(i), b(i), ip(i)});
  auto pt = point_ns.unchecked<1>();
  auto st = states.unchecked<2>();
  if (st.shape(1) != 10) throw std::invalid_argument("states must have 10 columns");
  for (py::ssize_t i = 0; i < pt.shape(0); ++i) {
    ScoreState s = make_state({st(i, 0), st(i, 1)}, {st(i, 2), st(i, 3)}, {st(i, 4), st(i, 5)},
                              st(i, 6), st(i, 7) != 0, st(i, 8), st(i, 9));
    m.timeline.push_back({pt(i), s});
  }
  return m;
}

BacktestConfig make_config(double delay_s, double threshold, int slippage_ticks, double min_gap_s,
                           int max_bets, double min_odds, double max_odds, bool trade_preplay,
                           const std::string& rule, double stake, double bankroll,
                           double kelly_fraction, double kelly_cap) {
  BacktestConfig c;
  c.delay_s = delay_s;
  c.threshold = threshold;
  c.slippage_ticks = slippage_ticks;
  c.min_gap_s = min_gap_s;
  c.max_bets = max_bets;
  c.min_odds = min_odds;
  c.max_odds = max_odds;
  c.trade_preplay = trade_preplay;
  if (rule == "flat") c.staking.rule = StakingConfig::Rule::Flat;
  else if (rule == "kelly") c.staking.rule = StakingConfig::Rule::Kelly;
  else throw std::invalid_argument("unknown staking rule: " + rule);
  c.staking.stake = stake;
  c.staking.bankroll = bankroll;
  c.staking.kelly_fraction = kelly_fraction;
  c.staking.kelly_cap = kelly_cap;
  return c;
}

// Flatten one MatchResult into numpy arrays (cheap to turn into DataFrames).
py::dict to_dict(const MatchResult& r) {
  const auto n = static_cast<py::ssize_t>(r.predictions.size());
  py::array_t<std::int64_t> time(n);
  py::array_t<double> market(n), game(n), set(n), match(n);
  py::array_t<int> known(n);
  for (py::ssize_t i = 0; i < n; ++i) {
    const auto& p = r.predictions[static_cast<size_t>(i)];
    time.mutable_at(i) = p.time_ns;
    market.mutable_at(i) = p.market_p;
    game.mutable_at(i) = p.model.game;
    set.mutable_at(i) = p.model.set;
    match.mutable_at(i) = p.model.match;
    known.mutable_at(i) = p.known_points;
  }
  py::list bets;
  for (const auto& b : r.bets)
    bets.append(py::make_tuple(b.time_ns, b.side, b.odds, b.stake, b.model_p, b.market_p));
  const auto& s = r.settlement;
  py::dict d;
  d["time_ns"] = time;
  d["market_p"] = market;
  d["model_game"] = game;
  d["model_set"] = set;
  d["model_match"] = match;
  d["known_points"] = known;
  d["bets"] = bets;
  d["settlement"] = py::dict(py::arg("n_bets") = s.n_bets, py::arg("staked") = s.staked,
                             py::arg("gross") = s.gross, py::arg("commission") = s.commission,
                             py::arg("net") = s.net);
  return d;
}
}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "CourtEdge C++ core: score state machine, Markov engine, Monte Carlo";
  m.def("version", [] { return std::string(version()); });

  py::class_<MatchFormat>(m, "MatchFormat")
      .def(py::init(&make_format), py::arg("best_of") = 3, py::arg("final_set") = "tiebreak")
      .def_readonly("best_of", &MatchFormat::best_of)
      .def_property_readonly("final_set",
                             [](const MatchFormat& f) { return std::string(to_string(f.final_set)); })
      .def_property_readonly("sets_to_win", &MatchFormat::sets_to_win);

  py::class_<ScoreState>(m, "ScoreState")
      .def(py::init(&make_state), py::arg("sets") = std::array<int, 2>{0, 0},
           py::arg("games") = std::array<int, 2>{0, 0},
           py::arg("points") = std::array<int, 2>{0, 0}, py::arg("server") = 0,
           py::arg("tiebreak") = false, py::arg("tiebreak_first_server") = 0,
           py::arg("winner") = -1)
      .def_readonly("sets", &ScoreState::sets)
      .def_readonly("games", &ScoreState::games)
      .def_readonly("points", &ScoreState::points)
      .def_readonly("server", &ScoreState::server)
      .def_readonly("tiebreak", &ScoreState::tiebreak)
      .def_readonly("tiebreak_first_server", &ScoreState::tiebreak_first_server)
      .def_readonly("winner", &ScoreState::winner)
      .def_property_readonly("is_over", &ScoreState::is_over)
      .def(py::self == py::self)
      .def("__repr__", &describe);

  m.def("advance", &advance, py::arg("state"), py::arg("point_winner"), py::arg("fmt"));

  py::class_<WinProbabilities>(m, "WinProbabilities")
      .def_readonly("game", &WinProbabilities::game)
      .def_readonly("set", &WinProbabilities::set)
      .def_readonly("match", &WinProbabilities::match);

  py::class_<MarkovEngine>(m, "MarkovEngine")
      .def(py::init<double, double, MatchFormat>(), py::arg("pa"), py::arg("pb"), py::arg("fmt"))
      .def("win_probabilities", &MarkovEngine::win_probabilities, py::arg("state"))
      .def("p_match", &MarkovEngine::p_match, py::arg("first_server") = 0);

  m.def("p_game", &p_game, py::arg("p"));
  m.def(
      "simulate",
      [](const ScoreState& s, double pa, double pb, const MatchFormat& f, std::int64_t n,
         std::uint64_t seed, unsigned threads) {
        py::gil_scoped_release release;  // worker threads never touch Python objects
        return simulate(s, pa, pb, f, {n, seed, threads});
      },
      py::arg("state"), py::arg("pa"), py::arg("pb"), py::arg("fmt"), py::arg("n") = 100'000,
      py::arg("seed") = 0, py::arg("threads") = 0);

  py::class_<MatchData>(m, "MatchData")
      .def(py::init(&make_match), py::arg("tick_ns"), py::arg("odds_a"), py::arg("odds_b"),
           py::arg("in_play"), py::arg("point_ns"), py::arg("states"), py::arg("pa"),
           py::arg("pb"), py::arg("fmt"), py::arg("first_server"), py::arg("winner"),
           py::arg("commission"));

  py::class_<BacktestConfig>(m, "BacktestConfig")
      .def(py::init(&make_config), py::arg("delay_s"), py::arg("threshold"),
           py::arg("slippage_ticks"), py::arg("min_gap_s"), py::arg("max_bets"),
           py::arg("min_odds"), py::arg("max_odds"), py::arg("trade_preplay"), py::arg("rule"),
           py::arg("stake"), py::arg("bankroll"), py::arg("kelly_fraction"),
           py::arg("kelly_cap"));

  m.def(
      "run_many",
      [](const std::vector<MatchData>& matches, const BacktestConfig& cfg, unsigned threads) {
        std::vector<MatchResult> results;
        {
          py::gil_scoped_release release;
          results = run_many(matches, cfg, threads);
        }
        py::list out;
        for (const auto& r : results) out.append(to_dict(r));
        return out;
      },
      py::arg("matches"), py::arg("cfg"), py::arg("threads") = 0);
  m.def("tick_down", &tick_down, py::arg("price"), py::arg("ticks") = 1);
}
