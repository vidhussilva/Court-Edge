// Python bindings: `from courtedge import _core`.
// Optional. The pure-Python package works without them; tests check parity when built.

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

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
}
