#include <gtest/gtest.h>

#include "courtedge/version.hpp"

TEST(Version, IsSemver) {
  const auto v = courtedge::version();
  EXPECT_EQ(v, "0.1.0");
}
