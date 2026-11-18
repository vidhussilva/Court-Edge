cmake_minimum_required(VERSION 3.24)
project(courtedge LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(COURTEDGE_BUILD_TESTS "Build GoogleTest suite" ON)
option(COURTEDGE_BUILD_BENCH "Build Google Benchmark targets" ON)
option(COURTEDGE_BUILD_PYTHON "Build the courtedge._core Python extension" OFF)

include(FetchContent)

add_library(courtedge_core
  src/version.cpp
  src/score.cpp
  src/markov.cpp
  src/simulate.cpp
)
target_include_directories(courtedge_core PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(courtedge_core PUBLIC Threads::Threads)
set_target_properties(courtedge_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(courtedge_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>)

if(COURTEDGE_BUILD_TESTS)
  # Prefer a system GoogleTest (apt: libgtest-dev); fall back to fetching it.
  find_package(GTest 1.14 QUIET)
  if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  enable_testing()
  add_executable(courtedge_tests
    tests/test_version.cpp
    tests/test_score.cpp
    tests/test_markov.cpp
  )
  target_link_libraries(courtedge_tests PRIVATE courtedge_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(courtedge_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

if(COURTEDGE_BUILD_BENCH)
  find_package(benchmark QUIET)
  if(NOT benchmark_FOUND)
    FetchContent_Declare(benchmark
      URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
  endif()
  add_executable(courtedge_bench bench/bench_markov.cpp)
  target_link_libraries(courtedge_bench PRIVATE courtedge_core benchmark::benchmark_main)
endif()

if(COURTEDGE_BUILD_PYTHON)
  # pybind11 comes from the uv dev environment: `uv run cmake ...` puts it on the path.
  find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core python/bindings.cpp)
  target_link_libraries(_core PRIVATE courtedge_core)
  # Drop the module straight into the package so `from courtedge import _core` works.
  set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/courtedge)
endif()
