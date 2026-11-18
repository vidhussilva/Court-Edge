import numpy as np
import pandas as pd
import pytest

from courtedge.report.metrics import (
    bootstrap_by_group,
    brier,
    calibration_table,
    expected_calibration_error,
    log_loss,
)


def test_brier_and_log_loss_hand_values():
    p, y = np.array([0.8, 0.3]), np.array([1, 0])
    assert brier(p, y) == pytest.approx((0.04 + 0.09) / 2)
    assert log_loss(p, y) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)


def test_perfect_calibration_has_zero_ece():
    p = np.repeat([0.25, 0.75], 400)
    y = np.r_[np.tile([1, 0, 0, 0], 100), np.tile([1, 1, 1, 0], 100)]
    assert expected_calibration_error(p, y) == pytest.approx(0.0)
    assert calibration_table(p, y)["n"].sum() == 800


def test_bootstrap_resamples_groups():
    df = pd.DataFrame({"g": np.repeat(np.arange(30), 10), "x": np.repeat(np.arange(30.0), 10)})
    est, lo, hi = bootstrap_by_group(df, "g", lambda d: d.x.mean(), n=500, seed=1)
    assert est == pytest.approx(14.5) and lo < est < hi
    # Group-level resampling: the CI reflects 30 groups, not 300 rows.
    assert hi - lo > 4
