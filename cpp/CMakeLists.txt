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

include(FetchContent)

add_library(courtedge_core
  src/version.cpp
)
target_include_directories(courtedge_core PUBLIC include)
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
  )
  target_link_libraries(courtedge_tests PRIVATE courtedge_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(courtedge_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
